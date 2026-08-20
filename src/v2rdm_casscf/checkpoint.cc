/*
 *  @BEGIN LICENSE
 *
 *  Hilbert: a space for quantum chemistry plugins to Psi4
 *
 *  Copyright (c) 2020 by its authors (LICENSE).
 *
 *  The copyrights for code used from other parties are included in
 *  the corresponding files.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 *  @END LICENSE
 */

#include <psi4/libmints/matrix.h>
#include <psi4/libmints/molecule.h>
#include <psi4/libmints/vector.h>
#include <psi4/libmints/wavefunction.h>
#include <psi4/libmints/writer_file_prefix.h>
#include <psi4/liboptions/liboptions.h>
#include <psi4/libpsi4util/PsiOutStream.h>
#include <psi4/libqt/qt.h>
#include <psi4/libtrans/integraltransform.h>
#include <psi4/libtrans/mospace.h>
#include <psi4/psi4-dec.h>

#include "v2rdm_solver.h"

#include <misc/omp.h>

#include <bpsdp_solver.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits.h>
#include <sstream>
#include <unistd.h>
#include <vector>

using namespace psi;

namespace hilbert {

namespace {

bool file_exists(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  return input.good();
}

std::string errno_message(int err) { return std::string(std::strerror(err)); }

std::string absolute_path(const std::string &path) {
  if (!path.empty() && path[0] == '/') {
    return path;
  }

  char cwd[PATH_MAX];
  if (getcwd(cwd, sizeof(cwd)) == nullptr) {
    throw PsiException("Could not determine current working directory while "
                       "staging checkpoint file.",
                       __FILE__, __LINE__);
  }
  return std::string(cwd) + "/" + path;
}

std::string file_size_string(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input.good()) {
    return "unknown";
  }
  std::streamoff size = input.tellg();
  if (size < 0) {
    return "unknown";
  }
  std::ostringstream ss;
  ss << size << " bytes";
  return ss.str();
}

void copy_binary_file(const std::string &source,
                      const std::string &destination) {
  if (source == destination) {
    return;
  }
  std::ifstream input(source, std::ios::binary);
  if (!input.good()) {
    throw PsiException("Could not open checkpoint source file: " + source,
                       __FILE__, __LINE__);
  }
  std::ofstream output(destination, std::ios::binary | std::ios::trunc);
  if (!output.good()) {
    throw PsiException("Could not open checkpoint destination file: " +
                           destination,
                       __FILE__, __LINE__);
  }

  std::vector<char> buffer(8 * 1024 * 1024);
  std::streamoff copied = 0;
  while (input.good()) {
    input.read(buffer.data(), buffer.size());
    std::streamsize nread = input.gcount();
    if (nread > 0) {
      output.write(buffer.data(), nread);
      copied += nread;
    }
    if (!output.good()) {
      std::ostringstream ss;
      ss << "Failed while writing checkpoint file: " << destination << " after "
         << copied << " bytes copied from " << source
         << " (source size: " << file_size_string(source) << ")";
      throw PsiException(ss.str(), __FILE__, __LINE__);
    }
  }
  if (!input.eof()) {
    throw PsiException("Failed while reading checkpoint source file: " + source,
                       __FILE__, __LINE__);
  }

  output.flush();
  if (!output.good()) {
    throw PsiException("Failed while flushing checkpoint file: " + destination,
                       __FILE__, __LINE__);
  }
}

std::string psio_unit_filename(size_t unit) {
  std::shared_ptr<PSIO> psio(new PSIO());
  char *name = nullptr;
  psio->get_filename(unit, &name);
  if (name == nullptr) {
    throw PsiException("Could not determine PSIO checkpoint filename.",
                       __FILE__, __LINE__);
  }

  std::string filename = PSIOManager::shared_object()->get_file_path(unit);
  filename += name;
  filename += ".";
  filename += std::to_string(unit);
  std::free(name);
  return filename;
}

std::string stage_checkpoint_file(const std::string &source,
                                  const std::string &destination) {
  std::string absolute_source = absolute_path(source);

  errno = 0;
  if (::remove(destination.c_str()) != 0 && errno != ENOENT) {
    throw PsiException(
        "Could not clear previous PSIO checkpoint staging file: " +
            destination + " (" + errno_message(errno) + ")",
        __FILE__, __LINE__);
  }

  errno = 0;
  if (::link(absolute_source.c_str(), destination.c_str()) == 0) {
    return "hard link";
  }
  int link_errno = errno;

  errno = 0;
  if (::symlink(absolute_source.c_str(), destination.c_str()) == 0) {
    return "symbolic link";
  }
  int symlink_errno = errno;

  copy_binary_file(absolute_source, destination);
  std::ostringstream ss;
  ss << "copy (hard link unavailable: " << errno_message(link_errno)
     << "; symbolic link unavailable: " << errno_message(symlink_errno) << ")";
  return ss.str();
}

} // namespace

std::string v2RDMSolver::CheckpointFilename() const {
  std::string filename = options_.get_str("CHECKPOINT_FILE");
  if (!filename.empty()) {
    return filename;
  }
  if (reference_wavefunction_ && reference_wavefunction_->molecule()) {
    return get_writer_file_prefix(reference_wavefunction_->molecule()->name()) +
           ".v2rdm.chk";
  }
  return "v2rdm.chk";
}

void v2RDMSolver::ImportCheckpointFileIfRequested() {
  if (checkpoint_file_imported_) {
    return;
  }

  std::string source = options_.get_str("RESTART_FROM_CHECKPOINT_FILE");
  if (source.empty() || source == "DUMMY") {
    return;
  }

  if (!file_exists(source)) {
    throw PsiException("RESTART_FROM_CHECKPOINT_FILE does not exist: " + source,
                       __FILE__, __LINE__);
  }

  std::string scratch = psio_unit_filename(PSIF_V2RDM_CHECKPOINT);
  std::string staging_mode = stage_checkpoint_file(source, scratch);
  PSIOManager::shared_object()->mark_file_for_retention(scratch, true);
  checkpoint_file_imported_ = true;

  outfile->Printf("    Imported v2RDM checkpoint file: %s -> %s (%s)\n",
                  source.c_str(), scratch.c_str(), staging_mode.c_str());
}

void v2RDMSolver::ExportCheckpointFile() {
  std::string scratch = psio_unit_filename(PSIF_V2RDM_CHECKPOINT);
  std::string destination = CheckpointFilename();
  copy_binary_file(scratch, destination);
  outfile->Printf("    Wrote v2RDM checkpoint file: %s\n", destination.c_str());
}

void v2RDMSolver::WriteCheckpointFile() {

  // Update Ca_/Cb_
  // UpdateTransformationMatrix();

  std::shared_ptr<PSIO> psio(new PSIO());

  std::string scratch = psio_unit_filename(PSIF_V2RDM_CHECKPOINT);
  errno = 0;
  if (::remove(scratch.c_str()) != 0 && errno != ENOENT) {
    throw PsiException(
        "Could not clear PSIO checkpoint scratch file before writing: " +
            scratch + " (" + errno_message(errno) + ")",
        __FILE__, __LINE__);
  }

  psio->open(PSIF_V2RDM_CHECKPOINT, PSIO_OPEN_NEW);

  // mu
  double mu = sdp_->get_mu();
  psio->write_entry(PSIF_V2RDM_CHECKPOINT, "MU", (char *)(&mu), sizeof(double));

  // x
  psio->write_entry(PSIF_V2RDM_CHECKPOINT, "PRIMAL", (char *)x->pointer(),
                    n_primal_ * sizeof(double));

  // y
  // psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL
  // 1",(char*)sdp_->get_y()->pointer(),n_dual_*sizeof(double));
  psio->write_entry(PSIF_V2RDM_CHECKPOINT, "DUAL 1", (char *)sdp_->get_y(),
                    n_dual_ * sizeof(double));

  // z
  // psio->write_entry(PSIF_V2RDM_CHECKPOINT,"DUAL
  // 2",(char*)sdp_->get_z()->pointer(),n_primal_*sizeof(double));
  psio->write_entry(PSIF_V2RDM_CHECKPOINT, "DUAL 2", (char *)sdp_->get_z(),
                    n_primal_ * sizeof(double));

  // mo/mo' transformation matrix
  psio_address addr = PSIO_ZERO;
  for (int h = 0; h < nirrep_; h++) {
    if (nmopi_[h] == 0)
      continue;
    double **np = newMO_->pointer(h);
    psio->write(PSIF_V2RDM_CHECKPOINT, "MO TO MO' TRANSFORMATION MATRIX",
                (char *)&(np[0][0]), nmopi_[h] * nmopi_[h] * sizeof(double),
                addr, &addr);
  }
  // so/mo transformation matrix
  addr = PSIO_ZERO;
  for (int h = 0; h < nirrep_; h++) {
    if (nsopi_[h] == 0 || nmopi_[h] == 0)
      continue;
    double **cp = Ca_->pointer(h);
    psio->write(PSIF_V2RDM_CHECKPOINT, "SO TO MO TRANSFORMATION MATRIX",
                (char *)&(cp[0][0]), nsopi_[h] * nmopi_[h] * sizeof(double),
                addr, &addr);
  }

  psio->close(PSIF_V2RDM_CHECKPOINT, 1);

  ExportCheckpointFile();
}

void v2RDMSolver::ReadFromCheckpointFile() {

  ImportCheckpointFileIfRequested();

  std::shared_ptr<PSIO> psio(new PSIO());

  if (!psio->exists(PSIF_V2RDM_CHECKPOINT)) {
    return;
  }

  outfile->Printf("\n");
  outfile->Printf("    ==> Restarting from checkpoint file <==\n");

  psio->open(PSIF_V2RDM_CHECKPOINT, PSIO_OPEN_OLD);

  // mu
  double mu = 0.1;
  psio->read_entry(PSIF_V2RDM_CHECKPOINT, "MU", (char *)(&mu), sizeof(double));
  sdp_->set_mu(mu);

  try {
    // x
    psio->read_entry(PSIF_V2RDM_CHECKPOINT, "PRIMAL", (char *)x->pointer(),
                     n_primal_ * sizeof(double));

    // y
    std::shared_ptr<Vector> y(new Vector(n_dual_));
    psio->read_entry(PSIF_V2RDM_CHECKPOINT, "DUAL 1", (char *)y->pointer(),
                     n_dual_ * sizeof(double));
    sdp_->set_y(y->pointer());
    y.reset();

    // z
    std::shared_ptr<Vector> z(new Vector(n_primal_));
    psio->read_entry(PSIF_V2RDM_CHECKPOINT, "DUAL 2", (char *)z->pointer(),
                     n_primal_ * sizeof(double));
    sdp_->set_z(z->pointer());
    z.reset();
  } catch (const std::exception &e) {
    outfile->Printf("    [Warning] Checkpoint primal/dual vector size mismatch "
                    "(likely due to different active space size).\n");
    outfile->Printf("              Orbitals loaded from checkpoint, but "
                    "initializing RDMs to SCF guess.\n");
  }

  psio->close(PSIF_V2RDM_CHECKPOINT, 1);
}

void v2RDMSolver::ReadOrbitalsFromCheckpointFile() {

  ImportCheckpointFileIfRequested();

  std::shared_ptr<PSIO> psio(new PSIO());

  if (!psio->exists(PSIF_V2RDM_CHECKPOINT)) {
    return;
  }

  psio->open(PSIF_V2RDM_CHECKPOINT, PSIO_OPEN_OLD);

  // Read the optimized orbitals (SO to MO transformation matrix) from the
  // checkpoint
  SharedMatrix tempCa(new Matrix(Ca_));
  psio_address addr = PSIO_ZERO;
  for (int h = 0; h < nirrep_; h++) {
    if (nsopi_[h] == 0 || nmopi_[h] == 0)
      continue;
    double **tp = tempCa->pointer(h);
    psio->read(PSIF_V2RDM_CHECKPOINT, "SO TO MO TRANSFORMATION MATRIX",
               (char *)&(tp[0][0]), nsopi_[h] * nmopi_[h] * sizeof(double),
               addr, &addr);
  }

  // Set Ca_ and Cb_ directly to the optimized orbitals
  Ca_->copy(tempCa);
  Cb_->copy(tempCa);

  // Reset newMO_ to the identity matrix
  newMO_->zero();
  for (int h = 0; h < nirrep_; h++) {
    for (int i = 0; i < nmopi_[h]; i++) {
      newMO_->pointer(h)[i][i] = 1.0;
    }
  }

  psio->close(PSIF_V2RDM_CHECKPOINT, 1);
}

std::string v2RDMSolver::QmoFilename() const {
  return CheckpointFilename() + ".qmo";
}

void v2RDMSolver::WriteQmoFile() {
  if (!is_df_ || Qmo_ == nullptr) return;

  long int nn1mo = (long int)nmo_ * ((long int)nmo_ + 1) / 2;
  long int nelem = (long int)nQ_ * nn1mo;
  double size_gb = nelem * 8.0 / 1e9;

  std::string path = QmoFilename();
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f.good()) {
    outfile->Printf("    [Warning] Could not open Qmo companion file for writing: %s\n",
                    path.c_str());
    return;
  }

  // Write the header from explicit 8-byte locals: nmo_ is a 4-byte int
  // (inherited from psi::Wavefunction), so writing sizeof(long int) directly
  // from &nmo_ would append 4 bytes of adjacent garbage.
  const long int hdr_nQ  = (long int)nQ_;
  const long int hdr_nmo = (long int)nmo_;
  f.write(reinterpret_cast<const char *>(&hdr_nQ),  sizeof(long int));
  f.write(reinterpret_cast<const char *>(&hdr_nmo), sizeof(long int));
  f.write(reinterpret_cast<const char *>(Qmo_),  nelem * sizeof(double));
  f.flush();

  if (!f.good()) {
    outfile->Printf("    [Warning] Failed while writing Qmo companion file: %s\n",
                    path.c_str());
    return;
  }

  outfile->Printf("    Wrote Qmo companion file: %s (%.2f GB)\n",
                  path.c_str(), size_gb);
}

bool v2RDMSolver::ReadQmoFromFile() {
  if (!is_df_) return false;

  // The qmo companion sits next to its checkpoint. On restart the matching qmo
  // was written by the previous run next to RESTART_FROM_CHECKPOINT_FILE, which
  // may differ from CHECKPOINT_FILE (this run's write target). Prefer the
  // restart-source path, then fall back to the write-target path.
  std::vector<std::string> candidates;
  std::string restart = options_.get_str("RESTART_FROM_CHECKPOINT_FILE");
  if (!restart.empty() && restart != "DUMMY") {
    candidates.push_back(restart + ".qmo");
  }
  candidates.push_back(QmoFilename());

  std::string path;
  std::ifstream f;
  for (const std::string &cand : candidates) {
    f.open(cand, std::ios::binary);
    if (f.good()) {
      path = cand;
      break;
    }
    f.clear();
  }
  if (path.empty()) {
    outfile->Printf("    [Warning] Qmo companion file not found (looked for:");
    for (const std::string &cand : candidates) {
      outfile->Printf(" %s", cand.c_str());
    }
    outfile->Printf(") — recomputing integrals.\n");
    return false;
  }

  long int file_nQ = 0, file_nmo = 0;
  f.read(reinterpret_cast<char *>(&file_nQ),  sizeof(long int));
  f.read(reinterpret_cast<char *>(&file_nmo), sizeof(long int));

  // Legacy qmo files (written before the header fix) stored nmo from a 4-byte
  // int with garbage in the upper 32 bits; the integral data itself is intact
  // at the fixed 16-byte offset. nmo always fits in 32 bits, so validate on the
  // low word — this accepts both correctly-written and legacy headers.
  const long int file_nmo_lo = file_nmo & 0xFFFFFFFFL;

  if (!f.good() || file_nQ != (long int)nQ_ || file_nmo_lo != (long int)nmo_) {
    outfile->Printf(
        "    [Warning] Qmo companion file dimension mismatch "
        "(file: nQ=%ld nmo=%ld, expected: nQ=%ld nmo=%ld) — recomputing integrals.\n",
        file_nQ, file_nmo_lo, (long int)nQ_, (long int)nmo_);
    return false;
  }

  long int nn1mo = (long int)nmo_ * ((long int)nmo_ + 1) / 2;
  long int nelem = (long int)nQ_ * nn1mo;

  Qmo_ = (double *)malloc(nelem * sizeof(double));
  if (Qmo_ == nullptr) {
    outfile->Printf("    [Warning] Could not allocate memory for Qmo — recomputing integrals.\n");
    return false;
  }

  f.read(reinterpret_cast<char *>(Qmo_), nelem * sizeof(double));
  if (!f.good()) {
    outfile->Printf("    [Warning] Failed reading Qmo companion file — recomputing integrals.\n");
    free(Qmo_);
    Qmo_ = nullptr;
    return false;
  }

  outfile->Printf("    Loaded Qmo from companion file: %s (%.2f GB)\n",
                  path.c_str(), nelem * 8.0 / 1e9);
  return true;
}

} // namespace hilbert
