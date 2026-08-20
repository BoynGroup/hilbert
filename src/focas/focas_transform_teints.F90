 ! 
 !  @BEGIN LICENSE
 ! 
 !  Hilbert: a space for quantum chemistry plugins to Psi4 
 ! 
 !  Copyright (c) 2020 by its authors (LICENSE).
 ! 
 !  The copyrights for code used from other parties are included in
 !  the corresponding files.
 ! 
 !  This program is free software: you can redistribute it and/or modify
 !  it under the terms of the GNU Lesser General Public License as published by
 !  the Free Software Foundation, either version 3 of the License, or
 !  (at your option) any later version.
 ! 
 !  This program is distributed in the hope that it will be useful,
 !  but WITHOUT ANY WARRANTY; without even the implied warranty of
 !  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 !  GNU Lesser General Public License for more details.
 ! 
 !  You should have received a copy of the GNU Lesser General Public License
 !  along with this program.  If not, see http://www.gnu.org/licenses/.
 ! 
 !  @END LICENSE
 ! 

module focas_transform_teints
  
  use focas_data
  use iso_c_binding, only: c_int, c_long_long, c_double

  implicit none

  interface
    integer(c_int) function hilbert_focas_df_c1_cuda_transform(nmo,nQ,int2,u,block_q,max_devices,verbose) &
        bind(C,name="hilbert_focas_df_c1_cuda_transform")
      import :: c_int, c_long_long, c_double
      integer(c_int), value       :: nmo
      integer(c_long_long), value :: nQ
      real(c_double)              :: int2(*)
      real(c_double), intent(in)  :: u(*)
      integer(c_int), value       :: block_q
      integer(c_int), value       :: max_devices
      integer(c_int), value       :: verbose
    end function hilbert_focas_df_c1_cuda_transform
  end interface
  
  contains

    integer function transform_teints_df(int2)
      implicit none

      real(wp) :: int2(:)

      type sym_R_info
        type(matrix_block), allocatable :: sym_L(:)
      end type sym_R_info

      type tmp_matrix
        real(wp), allocatable :: tmp(:,:)
        type(sym_R_info), allocatable :: sym_R(:)
      end type tmp_matrix

      type(tmp_matrix), allocatable :: aux(:)

      integer :: i_thread,sym_L,sym_R,L_eq_I,R_eq_I,L,R,max_nmopi
      integer :: nmo_R,nmo_L,R_copy
      integer :: nfz_R,nac_R,nfz_L,nac_L
      integer :: sym_cuda_status
      integer(ip) :: first_Q(nthread_use_),last_Q(nthread_use_)
      integer(ip) :: int_ind,nQ,Q
      logical :: profile_transform
      real(wp) :: t0(2),t1(2)
      real(wp) :: t_wall_alloc,t_wall_setup,t_wall_work,t_wall_dealloc
      real(wp) :: t_wall_gather,t_wall_transform,t_wall_scatter

      nQ                  = int(df_vars_%nQ,kind=ip)

      max_nmopi           = maxval(trans_%nmopi)

      profile_transform   = (log_print_ == 1)
      t_wall_alloc        = 0.0_wp
      t_wall_setup        = 0.0_wp
      t_wall_work         = 0.0_wp
      t_wall_dealloc      = 0.0_wp
      t_wall_gather       = 0.0_wp
      t_wall_transform    = 0.0_wp
      t_wall_scatter      = 0.0_wp

      if ( use_c1_blocked_df_transform() ) then
        transform_teints_df = transform_teints_df_c1_blocked()
        return
      end if

      ! symmetry-general GPU transform (reuses the dense-U CUDA kernel); falls
      ! through to the CPU transform below on any nonzero device status
      if ( use_sym_blocked_df_transform() ) then
        sym_cuda_status = transform_teints_df_sym_blocked()
        if ( sym_cuda_status == 0 ) then
          transform_teints_df = 0
          return
        end if
      end if

      if (profile_transform) t0 = timer()
      transform_teints_df = allocate_tmp_matrices()
      if (profile_transform) then
        t1 = timer()
        t_wall_alloc = t1(1) - t0(1)
      end if

      if (profile_transform) t0 = timer()
      transform_teints_df = setup_Q_bounds()
      if (profile_transform) then
        t1 = timer()
        t_wall_setup = t1(1) - t0(1)
      end if

      if (profile_transform) t0 = timer()
!$omp parallel shared(first_Q,last_Q,int2,df_vars_,nirrep_,profile_transform) &
!$omp& reduction(+:t_wall_gather,t_wall_transform,t_wall_scatter) num_threads(nthread_use_)
!$omp do private(i_thread,Q,int_ind,sym_R,sym_L,nmo_R,nmo_L,L,R,R_eq_I,L_eq_I,R_copy,t0,t1)

      do i_thread = 1 , nthread_use_

        ! loop over Q indeces to be transformed by this thread
 
        do Q = first_Q(i_thread) , last_Q(i_thread)

          ! *************************************************************
          ! *** GATHER ( only LT row > col elements are accessed in int2)
          ! *************************************************************

          if (profile_transform) t0 = timer()

!          int_ind =  int(Q,kind=ip)  
          int_ind = ( ( Q - 1 ) * int(ngem_tot_,kind = ip) ) + 1

          do sym_L = 1 , nirrep_

            nmo_L = trans_%nmopi(sym_L)

            do L = 1 , nmo_L

              do sym_R = 1 , sym_L - 1

                nmo_R = trans_%nmopi(sym_R)

                do R = 1 , nmo_R

                  aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(L,R)= int2(int_ind)

                  int_ind = int_ind + 1

                end do

              end do

              do R = 1 , L

                aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(R,L) = int2(int_ind)

                int_ind = int_ind + 1

              end do

            end do

          end do

          if (profile_transform) then
            t1 = timer()
            t_wall_gather = t_wall_gather + t1(1) - t0(1)
          end if

          ! **************************************************************
          ! *** TRANSFORM (only lower triangular blocks are transformed )
          ! **************************************************************

          if (profile_transform) t0 = timer()

          ! ***********************************************************
          ! THIS CODE ONLY TAKES ANDVANTAGE OF PARTIAL SPARSE STRUCTURE 
          ! WHEN EITHER L==I AND/OR R==I
          ! ***********************************************************

          do sym_R = 1 , nirrep_

            nmo_R    = trans_%nmopi(sym_R)

            R_eq_I   = trans_%U_eq_I(sym_R)

            if ( nmo_R == 0 ) cycle

            if ( R_eq_I == 0 ) then

              call symmetrize_diagonal_block(aux(i_thread)%sym_R(sym_R)%sym_L(sym_R)%val,nmo_R)

              call dgemm('N','N',nmo_R,nmo_R,nmo_R,1.0_wp,aux(i_thread)%sym_R(sym_R)%sym_L(sym_R)%val,nmo_R,&
              & trans_%u_irrep_block(sym_R)%val,nmo_R,0.0_wp,aux(i_thread)%tmp,max_nmopi)

              call dgemm('T','N',nmo_R,nmo_R,nmo_R,1.0_wp,trans_%u_irrep_block(sym_R)%val,nmo_R, &
              & aux(i_thread)%tmp,max_nmopi,0.0_wp,aux(i_thread)%sym_R(sym_R)%sym_L(sym_R)%val,nmo_R)

            end if

            do sym_L = sym_R +1 , nirrep_

              nmo_L    = trans_%nmopi(sym_L)

              if ( nmo_L == 0 ) cycle

              L_eq_I   = trans_%U_eq_I(sym_L)

              if ( L_eq_I == 1 ) then

                if ( R_eq_I /= 1 ) then

                  ! L == I and R /= I

                  call dgemm('n','n',nmo_L,nmo_R,nmo_R,1.0_wp,aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val,nmo_L,&
                  & trans_%u_irrep_block(sym_R)%val,nmo_R,0.0_wp,aux(i_thread)%tmp,max_nmopi)

                  do R_copy = 1 , nmo_R
                    call my_dcopy(nmo_L,aux(i_thread)%tmp(:,R_copy),1,&
                         & aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(:,R_copy),1)
                  end do

                end if

              else

                if ( R_eq_I == 1 ) then
                
                  ! L /= I and R == I

                  call dgemm('t','n',nmo_L,nmo_R,nmo_L,1.0_wp,trans_%u_irrep_block(sym_L)%val,nmo_L, &
                  & aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val,nmo_L,0.0_wp,aux(i_thread)%tmp,max_nmopi)

                  do R_copy = 1 , nmo_R
                    call my_dcopy(nmo_L,aux(i_thread)%tmp(:,R_copy),1,&
                         & aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(:,R_copy),1)
                  end do

                else

                  ! L /= I and R /= I

                  call dgemm('n','n',nmo_L,nmo_R,nmo_R,1.0_wp,aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val,nmo_L,&
                  & trans_%u_irrep_block(sym_R)%val,nmo_R,0.0_wp,aux(i_thread)%tmp,max_nmopi)

                  call dgemm('t','n',nmo_L,nmo_R,nmo_L,1.0_wp,trans_%u_irrep_block(sym_L)%val,nmo_L, &
                  & aux(i_thread)%tmp,max_nmopi,0.0_wp,aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val,nmo_L)

                end if

              end if

            end do ! sym_L loop

          end do ! sym_R loop

          if (profile_transform) then
            t1 = timer()
            t_wall_transform = t_wall_transform + t1(1) - t0(1)
          end if

          ! *************************************************************
          ! *** SCATTER (only LT row > col elements are accessed in int2)
          ! *************************************************************

          if (profile_transform) t0 = timer()

!          int_ind =  int(Q,kind=ip)
          int_ind = ( ( Q - 1 ) * int(ngem_tot_,kind = ip) ) + 1

          do sym_L = 1 , nirrep_

            nmo_L = trans_%nmopi(sym_L)

            do L = 1 , nmo_L

              do sym_R = 1 , sym_L - 1

                nmo_R = trans_%nmopi(sym_R)

                do R = 1 , nmo_R

                  int2(int_ind) = aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(L,R)

                  int_ind = int_ind + 1

                end do

              end do

              do R = 1 , L

                int2(int_ind) = aux(i_thread)%sym_R(sym_R)%sym_L(sym_L)%val(R,L) 

                int_ind = int_ind + 1

              end do

            end do

          end do

          if (profile_transform) then
            t1 = timer()
            t_wall_scatter = t_wall_scatter + t1(1) - t0(1)
          end if

        end do ! end Q loop

      end do ! end i_thread loop

!$omp end do
!$omp end parallel
      if (profile_transform) then
        t1 = timer()
        t_wall_work = t1(1) - t0(1)
      end if

      if (profile_transform) t0 = timer()
      transform_teints_df = deallocate_tmp_matrices()
      if (profile_transform) then
        t1 = timer()
        t_wall_dealloc = t1(1) - t0(1)
        write(fid_,'(a,1x,i10,7(1x,f11.3))') 'df_transform',int(nQ), &
             t_wall_alloc,t_wall_setup,t_wall_work,t_wall_gather, &
             t_wall_transform,t_wall_scatter,t_wall_dealloc
      end if

      return

      contains

        logical function use_c1_blocked_df_transform()

          implicit none

          integer           :: nmo

          use_c1_blocked_df_transform = .false.

          if ( focas_df_c1_blocked_enabled_ == 0 ) return
          if ( nirrep_ /= 1 ) return
          if ( df_vars_%Qstride /= ngem_tot_ ) return
          if ( .not. allocated(trans_%nmopi) ) return
          if ( .not. allocated(trans_%U_eq_I) ) return
          if ( trans_%nmopi(1) <= 0 ) return

          nmo = trans_%nmopi(1)
          if ( nmo * ( nmo + 1 ) / 2 /= ngem_tot_ ) return

          use_c1_blocked_df_transform = .true.

          return

        end function use_c1_blocked_df_transform

        integer function transform_teints_df_c1_blocked()

          implicit none

          integer                :: nmo,block_q,cuda_block_q,nQ_int
          integer                :: cuda_status
          integer                :: nrow_alloc,block_start,bq
          integer                :: q_rel,q_abs,row_offset
          integer                :: i_mo,j_mo
          integer(ip)            :: int_ind
          real(wp)               :: t_loc0(2),t_loc1(2)
          real(wp)               :: t_alloc,t_work,t_gather,t_transform
          real(wp)               :: t_scatter,t_dealloc
          real(wp)               :: scratch_budget_bytes,block_bytes
          real(wp), allocatable  :: right(:,:),tmp(:,:),left_mat(:,:),result(:,:)

          nmo      = trans_%nmopi(1)
          nQ_int   = int(nQ)

          if ( focas_df_c1_block_q_ > 0 ) then
            block_q = focas_df_c1_block_q_
          else
            block_bytes = 4.0_wp * 8.0_wp * real(nthread_use_,wp) * &
                 real(nmo,wp) * real(nmo,wp)
            if ( focas_df_c1_block_memory_mib_ > 0 ) then
              scratch_budget_bytes = real(focas_df_c1_block_memory_mib_,wp) * &
                   1024.0_wp * 1024.0_wp
            else
              scratch_budget_bytes = 64.0_wp * 1024.0_wp * 1024.0_wp * &
                   real(nthread_use_,wp)
            end if
            block_q = int(scratch_budget_bytes / block_bytes)
            block_q = max(1,min(focas_df_c1_block_q_max_,block_q))
          end if
          block_q  = min(block_q,nQ_int)

          transform_teints_df_c1_blocked = 0

          if ( trans_%U_eq_I(1) == 1 ) then
            if ( profile_transform ) then
              write(fid_,'(a,1x,i10,7(1x,f11.3),1x,a)') 'df_transform', &
                   int(nQ),0.0_wp,0.0_wp,0.0_wp,0.0_wp,0.0_wp,0.0_wp, &
                   0.0_wp,'c1_blocked_identity'
            end if
            return
          end if

          t_alloc     = 0.0_wp
          t_work      = 0.0_wp
          t_gather    = 0.0_wp
          t_transform = 0.0_wp
          t_scatter   = 0.0_wp
          t_dealloc   = 0.0_wp

          if ( focas_df_c1_cuda_enabled_ /= 0 ) then
            if ( focas_df_c1_cuda_validate_ /= 0 ) then
              if ( profile_transform ) then
                write(fid_,'(a)') 'df_transform_cuda validation requested; falling back to CPU transform'
              end if
            else
              if ( profile_transform ) t0 = timer()
              cuda_block_q = block_q
              if ( focas_df_c1_block_q_ <= 0 ) cuda_block_q = 0
              cuda_status = hilbert_focas_df_c1_cuda_transform( &
                   nmo,int(nQ_int,kind=c_long_long),int2, &
                   trans_%u_irrep_block(1)%val,cuda_block_q, &
                   focas_df_c1_cuda_num_gpus_, &
                   focas_df_c1_cuda_verbose_)
              if ( profile_transform ) then
                t1 = timer()
                t_work = t1(1) - t0(1)
              end if

              if ( cuda_status == 0 ) then
                if ( profile_transform ) then
                  write(fid_,'(a,1x,i10,7(1x,f11.3),1x,a,1x,i5)') &
                       'df_transform',int(nQ),0.0_wp,0.0_wp,t_work, &
                       0.0_wp,t_work,0.0_wp,0.0_wp,'c1_cuda',cuda_block_q
                end if
                return
              else
                if ( profile_transform ) then
                  write(fid_,'(a,1x,i6,1x,a)') &
                       'df_transform_cuda_fallback status',cuda_status, &
                       'using CPU c1_blocked transform'
                end if
              end if
            end if
          end if

          nrow_alloc = nmo * block_q

          if ( profile_transform ) t0 = timer()
!$omp parallel private(right,tmp,left_mat,result,block_start,bq,q_rel,q_abs, &
!$omp& row_offset,i_mo,j_mo,int_ind,t_loc0,t_loc1) &
!$omp& reduction(+:t_alloc,t_gather,t_transform,t_scatter,t_dealloc) &
!$omp& num_threads(nthread_use_)

          if ( profile_transform ) t_loc0 = timer()
          allocate(right(nrow_alloc,nmo))
          allocate(tmp(nrow_alloc,nmo))
          allocate(left_mat(nmo,nrow_alloc))
          allocate(result(nmo,nrow_alloc))
          if ( profile_transform ) then
            t_loc1 = timer()
            t_alloc = t_alloc + t_loc1(1) - t_loc0(1)
          end if

!$omp do schedule(dynamic)
          do block_start = 1 , nQ_int , block_q

            bq = min(block_q,nQ_int-block_start+1)

            if ( profile_transform ) t_loc0 = timer()
            do q_rel = 1 , bq
              q_abs      = block_start + q_rel - 1
              row_offset = ( q_rel - 1 ) * nmo
              int_ind    = ( ( int(q_abs,kind=ip) - 1_ip ) * &
                           int(ngem_tot_,kind=ip) ) + 1_ip

              do j_mo = 1 , nmo
                do i_mo = 1 , j_mo
                  right(row_offset+i_mo,j_mo) = int2(int_ind)
                  right(row_offset+j_mo,i_mo) = int2(int_ind)
                  int_ind = int_ind + 1_ip
                end do
              end do
            end do
            if ( profile_transform ) then
              t_loc1 = timer()
              t_gather = t_gather + t_loc1(1) - t_loc0(1)
            end if

            if ( profile_transform ) t_loc0 = timer()
            call dgemm('N','N',nmo*bq,nmo,nmo,1.0_wp,right,nrow_alloc, &
                 & trans_%u_irrep_block(1)%val,nmo,0.0_wp,tmp,nrow_alloc)

            do q_rel = 1 , bq
              row_offset = ( q_rel - 1 ) * nmo
              do j_mo = 1 , nmo
                left_mat(1:nmo,row_offset+j_mo) = &
                     tmp(row_offset+1:row_offset+nmo,j_mo)
              end do
            end do

            call dgemm('T','N',nmo,nmo*bq,nmo,1.0_wp, &
                 & trans_%u_irrep_block(1)%val,nmo,left_mat,nmo, &
                 & 0.0_wp,result,nmo)
            if ( profile_transform ) then
              t_loc1 = timer()
              t_transform = t_transform + t_loc1(1) - t_loc0(1)
            end if

            if ( profile_transform ) t_loc0 = timer()
            do q_rel = 1 , bq
              q_abs      = block_start + q_rel - 1
              row_offset = ( q_rel - 1 ) * nmo
              int_ind    = ( ( int(q_abs,kind=ip) - 1_ip ) * &
                           int(ngem_tot_,kind=ip) ) + 1_ip

              do j_mo = 1 , nmo
                do i_mo = 1 , j_mo
                  int2(int_ind) = result(i_mo,row_offset+j_mo)
                  int_ind = int_ind + 1_ip
                end do
              end do
            end do
            if ( profile_transform ) then
              t_loc1 = timer()
              t_scatter = t_scatter + t_loc1(1) - t_loc0(1)
            end if

          end do
!$omp end do

          if ( profile_transform ) t_loc0 = timer()
          deallocate(right)
          deallocate(tmp)
          deallocate(left_mat)
          deallocate(result)
          if ( profile_transform ) then
            t_loc1 = timer()
            t_dealloc = t_dealloc + t_loc1(1) - t_loc0(1)
          end if

!$omp end parallel
          if ( profile_transform ) then
            t1 = timer()
            t_work = t1(1) - t0(1)
            write(fid_,'(a,1x,i10,7(1x,f11.3),1x,a,1x,i5)') &
                 'df_transform',int(nQ),t_alloc,0.0_wp,t_work,t_gather, &
                 t_transform,t_scatter,t_dealloc,'c1_blocked',block_q
          end if

          return

        end function transform_teints_df_c1_blocked

        logical function use_sym_blocked_df_transform()
          ! Symmetry-general GPU DF transform. The existing dense-U CUDA kernel
          ! works for any point group: int2 is the full df-order (irrep-major)
          ! packed triangle, so transforming with a block-diagonal U reproduces
          ! U^T g U exactly (cross-irrep blocks are physically zero and stay so).
          implicit none

          use_sym_blocked_df_transform = .false.

          if ( focas_df_c1_cuda_enabled_ == 0 ) return
          if ( focas_df_c1_cuda_validate_ /= 0 ) return
          if ( df_vars_%Qstride /= ngem_tot_ ) return
          if ( nmo_tot_ <= 0 ) return
          if ( int(nmo_tot_,kind=ip) * ( int(nmo_tot_,kind=ip) + 1_ip ) / 2_ip /= &
               int(ngem_tot_,kind=ip) ) return
          if ( .not. allocated(trans_%nmopi) ) return
          if ( .not. allocated(trans_%U_eq_I) ) return

          use_sym_blocked_df_transform = .true.

          return
        end function use_sym_blocked_df_transform

        integer function transform_teints_df_sym_blocked()
          ! Assemble the block-diagonal orbital rotation U (df / irrep-major
          ! order) and hand the full packed int2 to the dense-U CUDA transform.
          implicit none

          integer               :: nmo,nQ_int,cuda_block_q,cuda_status
          integer               :: p_sym,i,j,base,all_identity
          real(wp)              :: t_work
          real(wp), allocatable :: u_dense(:,:)

          transform_teints_df_sym_blocked = 0

          nmo    = nmo_tot_
          nQ_int = int(nQ)

          ! if every irrep block is the identity, there is nothing to transform
          all_identity = 1
          do p_sym = 1 , nirrep_
            if ( trans_%nmopi(p_sym) <= 0 ) cycle
            if ( trans_%U_eq_I(p_sym) == 0 ) all_identity = 0
          end do
          if ( all_identity == 1 ) then
            if ( profile_transform ) &
              write(fid_,'(a,1x,i10,7(1x,f11.3),1x,a)') 'df_transform', &
                   int(nQ),0.0_wp,0.0_wp,0.0_wp,0.0_wp,0.0_wp,0.0_wp,0.0_wp, &
                   'sym_cuda_identity'
            return
          end if

          ! assemble the dense block-diagonal U in df (irrep-major) order
          allocate(u_dense(nmo,nmo))
          u_dense = 0.0_wp
          base = 0
          do p_sym = 1 , nirrep_
            if ( trans_%nmopi(p_sym) > 0 ) then
              if ( trans_%U_eq_I(p_sym) == 1 ) then
                do i = 1 , trans_%nmopi(p_sym)
                  u_dense(base+i,base+i) = 1.0_wp
                end do
              else
                do j = 1 , trans_%nmopi(p_sym)
                  do i = 1 , trans_%nmopi(p_sym)
                    u_dense(base+i,base+j) = trans_%u_irrep_block(p_sym)%val(i,j)
                  end do
                end do
              end if
            end if
            base = base + trans_%nmopi(p_sym)
          end do

          cuda_block_q = 0
          if ( focas_df_c1_block_q_ > 0 ) cuda_block_q = focas_df_c1_block_q_

          if ( profile_transform ) t0 = timer()
          cuda_status = hilbert_focas_df_c1_cuda_transform( &
               nmo,int(nQ_int,kind=c_long_long),int2,u_dense,cuda_block_q, &
               focas_df_c1_cuda_num_gpus_,focas_df_c1_cuda_verbose_)
          if ( profile_transform ) then
            t1 = timer()
            t_work = t1(1) - t0(1)
          end if

          deallocate(u_dense)

          if ( cuda_status /= 0 ) then
            transform_teints_df_sym_blocked = cuda_status
            if ( profile_transform .and. ( focas_df_c1_cuda_verbose_ /= 0 ) ) &
              write(fid_,'(a,1x,i6,1x,a)') &
                   'df_transform_sym_cuda_fallback status',cuda_status, &
                   'using CPU transform'
            return
          end if

          if ( profile_transform ) &
            write(fid_,'(a,1x,i10,7(1x,f11.3),1x,a,1x,i5)') 'df_transform', &
                 int(nQ),0.0_wp,0.0_wp,t_work,0.0_wp,t_work,0.0_wp,0.0_wp, &
                 'sym_cuda',cuda_block_q

          return
        end function transform_teints_df_sym_blocked

        subroutine symmetrize_diagonal_block(diag_block,ndim)

         implicit none

         ! simple function to symmetrize a square matrix for which onyl the LT elements 
         ! were stored

         integer, intent(in)     :: ndim
         real(wp), intent(inout) :: diag_block(ndim,ndim)

         integer :: R,L

         do R = 1 , ndim

           do L = R + 1 , ndim

             diag_block(L,R) = diag_block(R,L)

           end do

         end do

         return

        end subroutine symmetrize_diagonal_block


        integer function setup_Q_bounds()

          implicit none

          ! simple function to determine first and last auxiliary function index for each thread

          integer :: nQ_ave,i

          nQ_ave = df_vars_%nQ / nthread_use_

          do i = 1 , nthread_use_

            first_Q(i) = ( i - 1 ) * nQ_ave + 1
 
            last_Q(i)  = i * nQ_ave

          end do

          last_Q(nthread_use_) = df_vars_%nQ

          setup_Q_bounds = 0

          return

        end function setup_Q_bounds

        integer function allocate_tmp_matrices()

          implicit none

          ! simple function to allocate temporary matrices for 3-index integral transformation

          integer :: i,sym_L,sym_R,nmo_R,nmo_L
   
          allocate(aux(nthread_use_))        
 
          do i = 1 , nthread_use_

            allocate(aux(i)%tmp(max_nmopi,max_nmopi))

            allocate(aux(i)%sym_R(nirrep_))

            do sym_R = 1 , nirrep_

              nmo_R = trans_%nmopi(sym_R)

              if ( nmo_R == 0 ) cycle

              allocate(aux(i)%sym_R(sym_R)%sym_L(sym_R:nirrep_))

              do sym_L = sym_R , nirrep_

                nmo_L = trans_%nmopi(sym_L)
 
                if ( nmo_L == 0 ) cycle

                allocate(aux(i)%sym_R(sym_R)%sym_L(sym_L)%val(nmo_L,nmo_R))

              end do

            end do


          end do

          allocate_tmp_matrices = 0

          return

        end function allocate_tmp_matrices
     
        integer function deallocate_tmp_matrices()

          implicit none

          ! simple function to deallocate temporary matrices for 3-indes integral transformation

          integer :: i,sym_L,sym_R

          if ( .not. allocated(aux) ) return
 
          do i = 1 , size(aux)

            if ( allocated(aux(i)%tmp) )    deallocate(aux(i)%tmp)

            do sym_R = 1 , nirrep_

              if ( trans_%nmopi(sym_R) == 0 ) cycle

              do sym_L = sym_R , nirrep_

                if ( trans_%nmopi(sym_L) == 0 ) cycle

                deallocate(aux(i)%sym_R(sym_R)%sym_L(sym_L)%val)

              end do

              deallocate(aux(i)%sym_R(sym_R)%sym_L)

            end do

            deallocate(aux(i)%sym_R)

          end do

          deallocate(aux)

          deallocate_tmp_matrices = 0

          return

        end function deallocate_tmp_matrices

    end function transform_teints_df 

    integer function transform_teints(int2)
      implicit none

      ! driver to transform integrals g(ij|kl) --> g(rs|tu)
      ! ij irreps transformed sequentially

      real(wp) :: int2(:)

      integer     :: ij_sym
      integer     :: first_ij,last_ij
      integer     :: irrep_block_transformed(nirrep_)

      ! initialize error flag

      transform_teints = 1

      first_ij = ints_%offset(1) + 1
      last_ij  = first_ij + ints_%nnzpi(1) - 1

      irrep_block_transformed(1) = transform_teints_g0_block(int2(first_ij:last_ij),1)

      do ij_sym = 2 , nirrep_

        ! first/last nnz integral index in this irrep

        first_ij = ints_%offset(ij_sym) + 1
        last_ij  = first_ij + ints_%nnzpi(ij_sym) - 1

        ! transform integrals

        irrep_block_transformed(ij_sym) = transform_teints_irrep_block(int2(first_ij:last_ij),ij_sym)

        ! something went wrong, so return with error flag

        if ( irrep_block_transformed(ij_sym) /= 0 ) return

      end do

      ! if we got this far, the transformations are done

      transform_teints = 0

      return
    end function transform_teints


    integer function transform_teints_irrep_block(int_block,ij_sym)
      implicit none
      ! subroutine to transform an irrep-block (not totally symmetric) of two-electron integrals
      ! the algorithm performs the transformation in two half transfroms.
      ! 1) The first half transform consists of two steps:
      !    a) GATHER:    for a given ij pair, collect and store all integrals (ij|kl)
      !    b) TRANSFORM: for a given ij pair, perform two dense matrix-matrix multiplications to 
      !                   transform (ij|kl) --> (ij|tu)
      !
      ! To do the second half transform, half-transformed integrals need to be stored for ALL ij and tu
      ! This means, additional storage that is roughly twice the size of the LT integral block 
      !
      ! 2) The second half transform consists of three steps
      !    a) GATHER:    for a given tu pair, collect and store all integrals (ij|tu)
      !    c) TRANSFORM: for a given ij pair, perform two dense matrix-matrix multiplications to 
      !                  transform (ij|tu) --> (rs|tu)
      !    d) SCATTER:   scatter (rs|tu) into original integral array
      ! 
      ! NOTE: Currently, the GATHER and SCATTER operations do more work than the minimum required; however,
      !       after timing the total time spent in each type of operation, it is clear that the GATHER and
      !       SCATTER operations are an almost negligible fraction of the total time and thus, further
      !       optimization will be skipped. 
      integer, intent(in) :: ij_sym
      real(wp) :: int_block(:)

      type int_scr
        real(wp), allocatable :: mat(:,:)
      end type int_scr

      type int_tmp
        type(int_scr), allocatable :: irrep(:)
      end type int_tmp

      type index_info
        integer :: num_tu
        integer, allocatable :: tu(:,:)
      end type index_info

      type(index_info), allocatable :: tu_inds(:)

      type(int_tmp), allocatable :: A(:)

      real(wp), allocatable :: A_tilde(:,:),B_tilde(:,:)

      integer :: nnz_ij,max_nmopi

      integer :: i_class,j_class,k_class,l_class,t_class,u_class,tu_class,kl_class,ij_class
      integer :: i_irrep,j_irrep,k_irrep,l_irrep,t_irrep,u_irrep,tu
      integer :: i_sym,j_sym,k_sym,l_sym,t_sym,u_sym 
      integer :: num_i,num_j,num_k,num_l,num_t,num_u
      integer :: i_offset,j_offset,k_offset,l_offset,t_offset,u_offset

      integer :: ijkl_class,rstu_class

      ! initialize error flag

      transform_teints_irrep_block = 1
 
      ! the number of nnz integrals is not the same as the length of the vector int2

      if ( size(int_block,dim=1) /= ints_%nnzpi(ij_sym) ) return

      ! allocate temporary matrices

      ! maximum matrix size and number of matrices

      max_nmopi = maxval(trans_%nmopi)

      nnz_ij    = ints_%ngempi(ij_sym)

      ! allocate temporary arrays

      transform_teints_irrep_block = allocate_transform_scr()

      if ( transform_teints_irrep_block /= 0 ) call abort_print(311)

      ! ********************************
      ! *** 1ST HALF TRANSFORM (kl-->tu)
      ! ********************************

      ! loop over ij geminals

!$omp parallel shared(nnz_ij,nirrep_,ij_sym,trans_,int_block,A) num_threads(nthread_use_)
!$omp do private(ij_class,k_sym,l_sym,k_offset,l_offset,num_k,num_l,k_irrep,k_class,&
!$omp l_irrep,l_class,kl_class,ijkl_class,A_tilde)

      do ij_class = 1 , nnz_ij

        ! loop over symmetries for k

        do k_sym = 1 , nirrep_

          ! figure out symmetry of l such that ij_sym == kl_sym

          l_sym = group_mult_tab_(ij_sym,k_sym)

          ! make sure that we are only addressing unique integrals with k_sym > l_sym
 
          if ( k_sym < l_sym) cycle

          ! *******************************************
          ! *** GATHER *** A[ij][k_sym](k,l) = g(ij|kl)
          ! *******************************************

          ! here, we make use of the fact that this matrix has at most 8 nnz blocks,
          ! with the blocks indexed by k_sym

          ! offsets for figuring out full k & l indeces

          k_offset = trans_%offset(k_sym)
          l_offset = trans_%offset(l_sym)

          ! number of orbitals with symmetry k_sym and l_sym

          num_k    = trans_%nmopi(k_sym)
          num_l    = trans_%nmopi(l_sym) 

          ! loop over symmetry-reduced k indeces

          do k_irrep = 1 , num_k

            ! class k index for integral addressing

            k_class = trans_%irrep_to_class_map(k_irrep+k_offset)

            ! loop over symmetry reduced l indeces

            do l_irrep = 1 , num_l

              ! class l index for integral addressing

              l_class    = trans_%irrep_to_class_map(l_irrep+l_offset)

              ! integral address (class order)

              kl_class   = ints_%gemind(k_class,l_class)

              ijkl_class = pq_index(ij_class,kl_class) 

              ! save integral

              A(ij_class)%irrep(k_sym)%mat(k_irrep,l_irrep) = int_block(ijkl_class) 

            end do ! end l loop

          end do ! end k loop

          !*********************************************
          ! *** TRANSFORM A[ij] = C^T * A[ij][k_sym] * C
          !*********************************************

          ! A_tilde[ij][k_sym] = A[ij][k_sym] * C[l_sym]
          ! followed by
          ! A = (C[k_sym])^T * A_tilde[]ij][k_sym]

          ! only need to do work if there are orbitals with symmetry k_sym and l_sym 

          if ( ( num_k == 0 ) .or. ( num_l == 0 ) ) cycle

          call dgemm('N','N',num_k,num_l,num_l,1.0_wp,A(ij_class)%irrep(k_sym)%mat,num_k,&
                     trans_%u_irrep_block(l_sym)%val,num_l,0.0_wp,A_tilde,max_nmopi)

          call dgemm('T','N',num_k,num_l,num_k,1.0_wp,trans_%u_irrep_block(k_sym)%val,num_k, &
                     A_tilde,max_nmopi,0.0_wp,A(ij_class)%irrep(k_sym)%mat,num_k)
          
        end do ! end k_sym loop

      end do ! end ij_class loop      

!$omp end do
!$omp end parallel

      ! at this point, we have transformed the second index and 
      ! A[ij](t,u) = g(ij|tu)

      ! **********************
      ! *** 2ND HALF TRANSFORM
      ! **********************

      ! loop over symmetries for t

      do t_sym = 1 , nirrep_

        ! determine symmetry for u

        u_sym = group_mult_tab_(t_sym,ij_sym)

        ! make sure that we only address integrals with t_sym > u_sym

        if ( t_sym < u_sym ) cycle

        ! figure out number of orbitals with these symmetries

        num_t    = trans_%nmopi(t_sym)
        num_u    = trans_%nmopi(u_sym)

        ! this information is used in the scatter operation below

        t_offset = trans_%offset(t_sym)
        u_offset = trans_%offset(u_sym)

        ! loop over tu indeces

!$omp parallel shared(nirrep_,ij_sym,tu_inds,ints_,trans_,int_block,t_sym,u_sym, &
!$omp num_t,num_u,t_offset,u_offset,max_nmopi) num_threads(nthread_use_)
!$omp do private(tu,t_irrep,u_irrep,t_class,u_class,tu_class,i_sym,j_sym,num_i,num_j,i_offset,j_offset,     &
!$omp i_irrep,j_irrep,i_class,j_class,ij_class,B_tilde,A_tilde,rstu_class)

        do tu = 1 , tu_inds(t_sym)%num_tu

          ! retrieve irrep ordered t & u indeces

          t_irrep =  tu_inds(t_sym)%tu(1,tu)
          u_irrep =  tu_inds(t_sym)%tu(2,tu) 

          ! retrieve class ordered t & u indeces

          t_class =  trans_%irrep_to_class_map(t_offset+t_irrep)
          u_class =  trans_%irrep_to_class_map(u_offset+u_irrep)

          ! retrieve class ordered tu index

          tu_class = ints_%gemind(t_class,u_class)

          ! **********
          ! *** GATHER
          ! **********

          ! loop over symmetries for i

          do i_sym = 1 , nirrep_
 
            ! corresponding symmetry for j

            j_sym = group_mult_tab_(i_sym,ij_sym)

            ! make sure that we are only addressing unique integrals with i_sym > j_sym

            if ( i_sym < j_sym ) cycle

            ! number of i/j orbitals

            num_i = trans_%nmopi(i_sym)
            num_j = trans_%nmopi(j_sym)              

            ! offsets for indexing

            i_offset = trans_%offset(i_sym)
            j_offset = trans_%offset(j_sym)

            ! zero out temporary matrix

            B_tilde = 0.0_wp

            ! loop over i indeces

            do i_irrep = 1 , num_i

              ! i index in class order

              i_class = trans_%irrep_to_class_map(i_irrep+i_offset)

              do j_irrep = 1 , num_j
  
                ! j index in class order

                j_class      = trans_%irrep_to_class_map(j_irrep+j_offset)                
                  
                ij_class     = ints_%gemind(i_class,j_class)

                ! save the corresponding matrix element

                B_tilde(i_irrep,j_irrep) = A(ij_class)%irrep(t_sym)%mat(t_irrep,u_irrep)

              end do ! end j_irrep loop

            end do ! end i_irrep loop

            ! *************
            ! *** TRANSFORM
            ! *************
              
            ! A_tilde[tu][i_sym] = B_tilde[tu][i_sym] * C[[j_sym]
            ! followed by
            ! B_tilde[tu][i_sym] = (C[i_sym])^T * A_tilde[tu][i_sym]

            ! only need to do work if there are orbitals with symmetry i_sym and j_sym

            if ( ( num_i == 0 ) .or. ( num_j == 0 ) ) cycle 
 
            call dgemm('N','N',num_i,num_j,num_j,1.0_wp,B_tilde,max_nmopi,&
                   trans_%u_irrep_block(j_sym)%val,num_j,0.0_wp,A_tilde,max_nmopi)

            call dgemm('T','N',num_i,num_j,num_i,1.0_wp,trans_%u_irrep_block(i_sym)%val,num_i, &
                   A_tilde,max_nmopi,0.0_wp,B_tilde,max_nmopi)

            ! ***********
            ! *** SCATTER
            ! ***********

            ! at this point ij-->rs && kl-->tu so B_tilde[tu][i_sym]%mat(i,j) = g(rs|tu)

            ! loop over indeces for i

            do i_irrep = 1 , num_i

              ! i index in class order

              i_class = trans_%irrep_to_class_map(i_irrep+i_offset)

              do j_irrep = 1 , num_j

                ! j index in class order

                j_class               = trans_%irrep_to_class_map(j_irrep+j_offset)

                ! ij geminal index in class order

                ij_class              = ints_%gemind(i_class,j_class)

                ! integral address in class order

                if ( ij_class < tu_class ) cycle
 
                rstu_class            = pq_index(ij_class,tu_class)

                ! save integral

                int_block(rstu_class) = B_tilde(i_irrep,j_irrep)

              end do ! end j_irrep loop

            end do ! end i_irrep loop

          end do ! end i_sym loop

        end do ! end tu loop

!$omp end do
!$omp end parallel

      end do ! end t_sym loop

      ! ******** END ACTUAL WORK

      ! deallocate temporary arrays

      transform_teints_irrep_block = deallocate_transform_scr()

      return

      contains

        integer function allocate_transform_scr()

          integer :: ij,k,l,kl,k_sym,l_sym,num_k,num_l
          ! function to allocate scratch arrays for integral transformation

          allocate_transform_scr = 0 

          if (allocated(A)) allocate_transform_scr = deallocate_transform_scr()

          if ( allocate_transform_scr /= 0 ) call abort_print(312)

          allocate(A(nnz_ij))

          do ij = 1 , nnz_ij

            allocate(A(ij)%irrep(nirrep_))

            do k_sym = 1 , nirrep_

              l_sym = group_mult_tab_(k_sym,ij_sym)

              if ( k_sym < l_sym ) cycle

              num_k = trans_%nmopi(k_sym)
              num_l = trans_%nmopi(l_sym)

              ! do not allocate if there are no orbitals with these symmetries

              if ( ( num_k == 0 ) .or. ( num_l == 0 ) ) cycle         

              allocate(A(ij)%irrep(k_sym)%mat(num_k,num_l))

              A(ij)%irrep(k_sym)%mat = 0.0_wp

            end do

          end do
          
          if (allocated(A_tilde)) deallocate(A_tilde)

          allocate(A_tilde(max_nmopi,max_nmopi)) 

          if (allocated(B_tilde)) deallocate(B_tilde)

          allocate(B_tilde(max_nmopi,max_nmopi))

          if (allocated(tu_inds)) then

            do k_sym = 1 , nirrep_

              if ( .not. allocated(tu_inds(k_sym)%tu) ) cycle
 
              deallocate(tu_inds(k_sym)%tu)

            end do

            deallocate(tu_inds)

          endif

          allocate(tu_inds(nirrep_))

          do k_sym = 1 , nirrep_

            l_sym = group_mult_tab_(k_sym,ij_sym)
    
             tu_inds(k_sym)%num_tu = 0

            if ( k_sym < l_sym ) cycle

            num_k = trans_%nmopi(k_sym)
            num_l = trans_%nmopi(l_sym)

            tu_inds(k_sym)%num_tu = num_k * num_l

            if ( tu_inds(k_sym)%num_tu == 0 ) cycle

            allocate(tu_inds(k_sym)%tu(2,tu_inds(k_sym)%num_tu))

            kl = 0

            do k = 1 , num_k

              do l = 1 , num_l

                kl = kl + 1

                tu_inds(k_sym)%tu(1,kl)=k
                tu_inds(k_sym)%tu(2,kl)=l

              end do

            end do

          end do

          allocate_transform_scr = 0

          return
        end function allocate_transform_scr

        integer function deallocate_transform_scr()

          ! function to deallocate scratch arrays for integral transformation
          integer :: ij,k_sym

          deallocate_transform_scr = 1

          if (.not.allocated(A)) return

          do ij = 1 , nnz_ij
            
            if (.not.allocated(A(ij)%irrep)) cycle

            do k_sym = 1 , nirrep_

              if (allocated(A(ij)%irrep(k_sym)%mat)) deallocate(A(ij)%irrep(k_sym)%mat)

            end do

            deallocate(A(ij)%irrep)

          end do

          deallocate(A)

          if (allocated(A_tilde)) deallocate(A_tilde)

          if (allocated(B_tilde)) deallocate(B_tilde)

          if (allocated(tu_inds)) then

            do k_sym = 1 , nirrep_

              if ( .not. allocated(tu_inds(k_sym)%tu) ) cycle

              deallocate(tu_inds(k_sym)%tu)

            end do

            deallocate(tu_inds)

          endif

          deallocate_transform_scr = 0

          return

        end function deallocate_transform_scr

    end function transform_teints_irrep_block

    integer function transform_teints_g0_block(int_block,ij_sym)
      implicit none
      ! subroutine to transform an irrep-block (totally symmetric) of two-electron integrals
      ! the algorithm performs the transformation in two half transfroms.
      ! 1) The first half transform consists of two steps:
      !    a) GATHER:    for a given ij pair, collect and store all integrals (ij|kl)
      !    b) TRANSFORM: for a given ij pair, perform two dense matrix-matrix multiplications to 
      !                   transform (ij|kl) --> (ij|tu)
      !
      ! To do the second half transform, half-transformed integrals need to be stored for ALL ij and tu
      ! This means, additional storage that is roughly twice the size of the LT integral block 
      !
      ! 2) The second half transform consists of three steps
      !    a) GATHER:    for a given tu pair, collect and store all integrals (ij|tu)
      !    c) TRANSFORM: for a given ij pair, perform two dense matrix-matrix multiplications to 
      !                  transform (ij|tu) --> (rs|tu)
      !    d) SCATTER:   scatter (rs|tu) into original integral array
      ! 
      ! NOTE: Currently, the GATHER and SCATTER operations do more work than the minimum required; however,
      !       after timing the total time spent in each type of operation, it is clear that the GATHER and
      !       SCATTER operations are an almost negligible fraction of the total time and thus, further
      !       optimization will be skipped. 

      integer, intent(in) :: ij_sym
      real(wp) :: int_block(:)

      type int_scr
        real(wp), allocatable :: mat(:,:)
      end type int_scr

      type int_tmp
        type(int_scr), allocatable :: irrep(:)
      end type int_tmp

      type index_info
        integer :: num_tu
        integer, allocatable :: tu(:,:)
      end type index_info

      type(index_info), allocatable :: tu_inds(:)

      type(int_tmp), allocatable :: A(:)

      real(wp), allocatable :: A_tilde(:,:),B_tilde(:,:)

      integer :: nnz_ij,max_nmopi

      integer :: i_class,j_class,k_class,l_class,t_class,u_class,tu_class,tu_irrep,kl_class,ij_class
      integer :: i_irrep,j_irrep,k_irrep,l_irrep,t_irrep,u_irrep
      integer :: i_sym,k_sym,t_sym 
      integer :: num_i,num_k,num_t
      integer :: i_offset,k_offset,t_offset

      integer :: ijkl_class,rstu_class

      ! initialize error flag

      transform_teints_g0_block = 1
 
      ! the number of nnz integrals is not the same as the length of the vector int2

      if ( size(int_block,dim=1) /= ints_%nnzpi(ij_sym) ) return

      ! maximum matrix size and number of matrices

      max_nmopi = maxval(trans_%nmopi)

      nnz_ij    = ints_%ngempi(ij_sym)

      ! allocate temporary arrays

      transform_teints_g0_block = allocate_transform_scr_g0()

      if ( transform_teints_g0_block /= 0 ) call abort_print(313)

      ! ********************************
      ! *** 1ST HALF TRANSFORM (kl-->tu)
      ! ********************************

      ! loop over ij geminals

!$omp parallel shared(nnz_ij,max_nmopi,nirrep_,trans_,ints_,int_block,A) num_threads(nthread_use_)
!$omp do private(k_sym,k_offset,num_k,k_irrep,l_irrep,k_class,l_class,&
!$omp ij_class,kl_class,ijkl_class,A_tilde)

      do ij_class = 1 , nnz_ij

        ! loop over symmetries for k

        do k_sym = 1 , nirrep_

          ! figure out symmetry of l such that ij_sym == kl_sym

          ! *******************************************
          ! *** GATHER *** A[ij][k_sym](k,l) = g(ij|kl)
          ! *******************************************

          ! here, we make use of the fact that this matrix has at most 8 nnz blocks,
          ! with the blocks indexed by k_sym

          ! offsets for figuring out full k & l indeces

          k_offset = trans_%offset(k_sym)

          ! number of orbitals with symmetry k_sym 

          num_k    = trans_%nmopi(k_sym)

          ! loop over symmetry-reduced k indeces

          do k_irrep = 1 , num_k

            ! class k index for integral addressing

            k_class = trans_%irrep_to_class_map(k_irrep+k_offset)

            ! loop over symmetry reduced l indeces

            do l_irrep = 1 , k_irrep

              ! class l index for integral addressing

              l_class    = trans_%irrep_to_class_map(l_irrep+k_offset)

              ! integral address (class order)

              kl_class   = ints_%gemind(k_class,l_class)
              ijkl_class = pq_index(ij_class,kl_class) 

              ! save integral

              A(ij_class)%irrep(k_sym)%mat(k_irrep,l_irrep) = int_block(ijkl_class) 
              A(ij_class)%irrep(k_sym)%mat(l_irrep,k_irrep) = A(ij_class)%irrep(k_sym)%mat(k_irrep,l_irrep)

            end do ! end l loop

          end do ! end k loop

          !*********************************************
          ! *** TRANSFORM A[ij] = C^T * A[ij][k_sym] * C
          !*********************************************

          ! A_tilde[ij][k_sym] = A[ij][k_sym] * C[k_sym]
          ! followed by
          ! A = (C[k_sym])^T * A_tilde[]ij][k_sym]

          ! only need to do work if there are orbitals with symmetry k_sym  

          if ( num_k == 0 )  cycle

          call dgemm('N','N',num_k,num_k,num_k,1.0_wp,A(ij_class)%irrep(k_sym)%mat,num_k,&
                     trans_%u_irrep_block(k_sym)%val,num_k,0.0_wp,A_tilde,max_nmopi)

          call dgemm('T','N',num_k,num_k,num_k,1.0_wp,trans_%u_irrep_block(k_sym)%val,num_k, &
                     A_tilde,max_nmopi,0.0_wp,A(ij_class)%irrep(k_sym)%mat,num_k)

        end do ! end k_sym loop

      end do ! end ij_class loop      

!$omp end do
!$omp end parallel


      ! at this point, we have transformed the second index and 
      ! A[ij](t,u) = g(ij|tu)

      ! **********************
      ! *** 2ND HALF TRANSFORM
      ! **********************

      ! loop over symmetries for t

      do t_sym = 1 , nirrep_

        ! figure out number of orbitals with these symmetries

        num_t    = trans_%nmopi(t_sym)

        ! this information is used in the scatter operation below

        t_offset = trans_%offset(t_sym)
  
        ! loop over tu indeces

!$omp parallel shared(t_sym,num_t,t_offset,nirrep_,trans_,ints_,int_block,A,tu_inds,&
!$omp max_nmopi) num_threads(nthread_use_)
!$omp do private(tu_irrep,t_irrep,u_irrep,t_class,u_class, &
!$omp i_sym,num_i,i_offset,i_irrep,i_class,j_irrep,j_class,&
!$omp tu_class,ij_class,rstu_class,B_tilde,A_tilde)

        do tu_irrep = 1 , tu_inds(t_sym)%num_tu

          t_irrep = tu_inds(t_sym)%tu(1,tu_irrep)
          u_irrep = tu_inds(t_sym)%tu(2,tu_irrep)

          t_class = trans_%irrep_to_class_map(t_offset+t_irrep)
          u_class = trans_%irrep_to_class_map(t_offset+u_irrep) 

          tu_class = ints_%gemind(t_class,u_class)   

          ! **********
          ! *** GATHER
          ! **********

          ! loop over symmetries for i

          do i_sym = 1 , nirrep_

            num_i = trans_%nmopi(i_sym)

            ! offsets for indexing

            i_offset = trans_%offset(i_sym)

            ! zero out temporary matrix

            B_tilde = 0.0_wp

            ! loop over i indeces

            do i_irrep = 1 , num_i

              ! i index in class order

              i_class = trans_%irrep_to_class_map(i_irrep+i_offset)

              do j_irrep = 1 , i_irrep
  
                ! j index in class order

                j_class      = trans_%irrep_to_class_map(j_irrep+i_offset)                
                  
                ij_class     = ints_%gemind(i_class,j_class)

                ! save the corresponding matrix element

                B_tilde(i_irrep,j_irrep) = A(ij_class)%irrep(t_sym)%mat(t_irrep,u_irrep)
                B_tilde(j_irrep,i_irrep) = B_tilde(i_irrep,j_irrep)

              end do ! end j_irrep loop

            end do ! end i_irrep loop

            ! *************
            ! *** TRANSFORM
            ! *************

            ! A_tilde[tu][i_sym] = B_tilde[tu][i_sym] * C[[i_sym]
            ! followed by
            ! B_tilde[tu][i_sym] = (C[i_sym])^T * A_tilde[tu][i_sym]

            ! only need to do work if there are orbitals with symmetry k_sym  
 
            if ( num_i == 0 )  cycle

            call dgemm('N','N',num_i,num_i,num_i,1.0_wp,B_tilde,max_nmopi,&
                   trans_%u_irrep_block(i_sym)%val,num_i,0.0_wp,A_tilde,max_nmopi)

            call dgemm('T','N',num_i,num_i,num_i,1.0_wp,trans_%u_irrep_block(i_sym)%val,num_i, &
                   A_tilde,max_nmopi,0.0_wp,B_tilde,max_nmopi)

            ! ***********
            ! *** SCATTER
            ! ***********

            ! at this point ij-->rs && kl-->tu so B_tilde[tu][i_sym]%mat(i,j) = g(rs|tu)

            ! loop over indeces for i

            do i_irrep = 1 , num_i

              ! i index in class order

              i_class = trans_%irrep_to_class_map(i_irrep+i_offset)

              do j_irrep = 1 , i_irrep

                ! j index in class order

                j_class               = trans_%irrep_to_class_map(j_irrep+i_offset)

                ! ij geminal index in class order

                ij_class              = ints_%gemind(i_class,j_class)

                if ( ij_class < tu_class ) cycle

                ! integral address in class order

                rstu_class            = pq_index(ij_class,tu_class)

                ! save integral

                int_block(rstu_class) = B_tilde(i_irrep,j_irrep)


              end do ! end j_irrep loop

            end do ! end i_irrep loop

          end do ! end i_sym loop

        end do ! tu loop
      
!$omp end do
!$omp end parallel
    
      end do ! end t_sym loop

      ! ******** END ACTUAL WORK

      ! deallocate temporary arrays

      transform_teints_g0_block = deallocate_transform_scr_g0()

      return

      contains

        integer function allocate_transform_scr_g0()

          integer :: ij,k,l,kl,k_sym,num_k
          ! function to allocate scratch arrays for integral transformation

          allocate_transform_scr_g0 = 0 

          if (allocated(A)) allocate_transform_scr_g0 = deallocate_transform_scr_g0()

          if ( allocate_transform_scr_g0 /= 0 ) call abort_print(314)

          allocate(A(nnz_ij))

          do ij = 1 , nnz_ij

            allocate(A(ij)%irrep(nirrep_))

            do k_sym = 1 , nirrep_

              num_k = trans_%nmopi(k_sym)
          
              ! do not allocate if there are no orbitals with this symemtry
            
              if ( num_k == 0 ) cycle
 
              allocate(A(ij)%irrep(k_sym)%mat(num_k,num_k))

              A(ij)%irrep(k_sym)%mat = 0.0_wp

            end do

          end do
          
          if (allocated(A_tilde)) deallocate(A_tilde)

          allocate(A_tilde(max_nmopi,max_nmopi)) 

          if (allocated(B_tilde)) deallocate(B_tilde)

          allocate(B_tilde(max_nmopi,max_nmopi))

          if (allocated(tu_inds)) then

            do k_sym = 1 , nirrep_

              if ( .not. allocated(tu_inds(k_sym)%tu) ) cycle

              deallocate(tu_inds(k_sym)%tu)

            end do

            deallocate(tu_inds)

          endif

          allocate(tu_inds(nirrep_))

          do k_sym = 1 , nirrep_

            tu_inds(k_sym)%num_tu = 0

            num_k = trans_%nmopi(k_sym)

            tu_inds(k_sym)%num_tu = num_k * ( num_k + 1 ) / 2

            if ( tu_inds(k_sym)%num_tu == 0 ) cycle

            allocate(tu_inds(k_sym)%tu(2,tu_inds(k_sym)%num_tu))

            kl = 0

            do k = 1 , num_k

              do l = 1 , k

                kl = kl + 1

                tu_inds(k_sym)%tu(1,kl)=k
                tu_inds(k_sym)%tu(2,kl)=l

              end do

            end do

          end do

          allocate_transform_scr_g0 = 0

          return
        end function allocate_transform_scr_g0

        integer function deallocate_transform_scr_g0()

          ! function to deallocate scratch arrays for integral transformation
          integer :: ij,k_sym

          deallocate_transform_scr_g0 = 1

          if (.not.allocated(A)) return

          do ij = 1 , nnz_ij
            
            if (.not.allocated(A(ij)%irrep)) cycle

            do k_sym = 1 , nirrep_

              if (allocated(A(ij)%irrep(k_sym)%mat)) deallocate(A(ij)%irrep(k_sym)%mat)

            end do

            deallocate(A(ij)%irrep)

          end do

          deallocate(A)

          if (allocated(A_tilde)) deallocate(A_tilde)

          if (allocated(B_tilde)) deallocate(B_tilde)

          if (allocated(tu_inds)) then

            do k_sym = 1 , nirrep_

              if ( .not. allocated(tu_inds(k_sym)%tu) ) cycle

              deallocate(tu_inds(k_sym)%tu)

            end do

            deallocate(tu_inds)

          endif

          deallocate_transform_scr_g0 = 0

          return

        end function deallocate_transform_scr_g0

    end function transform_teints_g0_block

end module focas_transform_teints
