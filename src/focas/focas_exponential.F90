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

module focas_exponential

  use focas_data

  implicit none

  real(wp), parameter :: max_error_tolerance = 1.0e-10_wp

  contains

    subroutine compute_exponential(kappa_in)
      implicit none
      ! subroutine to compute matrix exponential of a skew-symmetric real matrix K
      ! U = exp(K)
      ! Since the matrix K is block-diagonal, U is as well. Thus, the matrix
      ! exponential is computed for each block separately
      
      integer :: i_sym,max_nmopi,error
      real(wp), intent(in) :: kappa_in(:)
      real(wp), allocatable :: k_block(:,:)

      ! maximum size of temporary matrix

      max_nmopi = maxval(trans_%nmopi)
 
      ! allocate temporary matrix

      allocate(k_block(max_nmopi,max_nmopi))

      if ( allocated(trans_%compact_rank) ) trans_%compact_rank = 0

      do i_sym=1,nirrep_

        error = gather_kappa_block(kappa_in,k_block,i_sym)
        if (error /= 0) call abort_print(10) 

        error = 0
        if ( trans_%nmopi(i_sym) > 0 ) error = compute_block_exponential(k_block,i_sym,max_nmopi)
        if (error /= 0) call abort_print(11)
         
      end do

      deallocate(k_block)

      return
 
    end subroutine compute_exponential

    integer function compute_block_exponential(K,block_sym,max_dim)
      implicit none
      ! function to compute the matrix exponential of a matrix according to
      ! U = exp(K) = X * cos(d) * X^T + K * X * d^(-1) * sin(d) * X^T
      ! where X and d are solutions of the eigenvalue equation K^2 * X = lambda * X
      ! and d = sqrt(-lambda)
      integer, intent(in)  :: block_sym,max_dim
      real(wp), intent(in) :: K(max_dim,max_dim)

      real(wp), allocatable :: K2(:,:),X(:,:),tmp_mat_1(:,:),tmp_mat_2(:,:)
      real(wp), allocatable :: d(:),work(:)
      integer, allocatable :: isuppz(:),iwork(:)
      integer :: iwork_tmp(1)
      real(wp) :: work_tmp(1,1),val

      integer :: block_dim,i,j, il,iu,neig_found,lwork,liwork,success,nfzc,nmo
      real(wp) :: vl,vu,diag_tol,max_normalization_error,max_orthogonality_error

      ! initialize return value
      compute_block_exponential = 1

      ! check to see if this block is going to be equal to the I

      if ( trans_%U_eq_I(block_sym) == 1 ) then

        if ( allocated(trans_%compact_rank) ) trans_%compact_rank(block_sym) = 0

        trans_%u_irrep_block(block_sym)%val = 0.0_wp

        do i = 1 , trans_%nmopi(block_sym)

          trans_%u_irrep_block(block_sym)%val(i,i) = 1.0_wp

        end do

        compute_block_exponential = 0

        return

      end if

      ! number of frozen doubly occupied orbitals
     
      nfzc = nfzcpi_(block_sym)

      ! save total number of orbitals

      nmo = trans_%nmopi(block_sym)

      ! dimension of active block

      block_dim = nmo - nfzc

      ! For the large-C1 use case, form an exact compact representation of
      ! U-I before considering the dense eigensolve.  The dense matrix U is
      ! still reconstructed for the existing OEI/MO-coefficient transforms
      ! and for every CPU fallback, while the CUDA DF transform consumes V
      ! and A directly.
      if ( focas_compact_rotation_enabled_ /= 0 .and. &
           allocated(trans_%compact_rank) .and. &
           allocated(trans_%compact_v) .and. allocated(trans_%compact_a) ) then
        if ( allocated(trans_%compact_v(block_sym)%val) .and. &
             allocated(trans_%compact_a(block_sym)%val) ) then
          success = compute_compact_block_exponential(K,block_sym,max_dim)
          if ( success == 0 ) then
            compute_block_exponential = 0
            return
          end if
          trans_%compact_rank(block_sym) = 0
        end if
      end if

      ! ***************************
      ! allocate temporary matrices
      ! ***************************

      allocate(K2(block_dim,block_dim))
      allocate(d(block_dim))
      allocate(X(block_dim,block_dim))

      ! ***************
      ! compute and K^2
      ! ***************

      do i = 1 , block_dim
        call my_dcopy(block_dim,K(nfzc+1:block_dim+nfzc,i+nfzc),1,X(:,i),1)
      end do

      call dgemm('n','n',block_dim,block_dim,block_dim,1.0_wp,K(nfzc+1:nmo,nfzc+1:nmo),& 
      & block_dim,X,block_dim,0.0_wp,K2,block_dim)

      ! ***************
      ! diagonalize K^2
      ! ***************
      
      X = K2
      call dsyev('v','u',block_dim,X,block_dim,d,work_tmp,-1,success)
      lwork=int(work_tmp(1,1))
      if ( success /= 0 ) then 
        deallocate(K2,d,X)
        return
      end if
      allocate(work(lwork))
      call dsyev('v','u',block_dim,X,block_dim,d, work,lwork,success)
      if ( success /= 0 ) then 
        deallocate(work)
        return
      endif

      ! *************************
      ! initialize unitary matrix
      ! *************************

      trans_%u_irrep_block(block_sym)%val = 0.0_wp

      do i = 1 , nfzc

        trans_%u_irrep_block(block_sym)%val(i,i) = 1.0_wp

      end do

      ! ***********************************************************************************
      ! compute exponential U = exp(K) = X * cos(d) * X^(T) + K * X * d^(-1) * sin(d) * X^T
      ! first compute the terms involving sin(d) followed by the terms involving cos(d)
      ! ***********************************************************************************

      ! scale eigenvalues
      do i = 1 , block_dim
        if (d(i) < 0.0_wp ) then 
          d(i) = sqrt(-d(i))
        else
          d(i) = sqrt(d(i))
        end if
      end do

      allocate(tmp_mat_1(block_dim,block_dim))
      allocate(tmp_mat_2(block_dim,block_dim))

      ! compute d^(-1) * sin(d) * X^T
      ! since both d and sin(d) are diagonal matrices, d^(-1) * sin(d) is also diagonal with diagonal elements sin(d(i))/d(i)
      ! since d^(-1) * sin(d) is diagonal, we can perform the matrix product C = D * M efficiently by recognizing that the ith
      ! row of C is just a scaled version of the ith row of M ... C(:,i) = D(i,i) * M(:,i) 
      ! note to self :: below, we are storing the transpose of d^(-1) * sin(d) * X^T since we directly copy rows --> rows

      do i = 1 , block_dim
        val = 1.0_wp
        if ( d(i) /= 0.0_wp ) val = sin(d(i)) / d(i) 

        call my_dcopy(block_dim,X(:,i),1,tmp_mat_1(:,i),1)
        call my_dscal(block_dim,val,tmp_mat_1(:,i),1)

      end do

      call dgemm('n','t',block_dim,block_dim,block_dim,1.0_wp,X,block_dim,tmp_mat_1, &
                & block_dim,0.0_wp,tmp_mat_2,block_dim)

      call dgemm('n','n',block_dim,block_dim,block_dim,1.0_wp,K(nfzc+1:nmo,nfzc+1:nmo),block_dim,tmp_mat_2, &
                & block_dim,0.0_wp,trans_%u_irrep_block(block_sym)%val(nfzc+1:nmo,nfzc+1:nmo),block_dim)
     
      ! ****************************
      ! *** COMPUTE X * cos(d) * X^T
      ! ****************************

      ! tmp_mat_1 =  cos(d) * X

      do i = 1 , block_dim
        val = cos(d(i))

        call my_dcopy(block_dim,X(:,i),1,tmp_mat_1(:,i),1)
        call my_dscal(block_dim,val,tmp_mat_1(:,i),1)        

      end do 

      call dgemm('n','t',block_dim,block_dim,block_dim,1.0_wp,X,block_dim,tmp_mat_1, & 
                & block_dim,1.0_wp,trans_%u_irrep_block(block_sym)%val(nfzc+1:nmo,nfzc+1:nmo),block_dim)

      max_orthogonality_error = 0.0_wp
      max_normalization_error = 0.0_wp

      do i = 1 , nmo

        ! compute norm of vector

        val = my_ddot(nmo,trans_%u_irrep_block(block_sym)%val(:,i),1,&
              trans_%u_irrep_block(block_sym)%val(:,i),1)

        if ( abs( 1.0_wp - val ) > max_normalization_error ) max_normalization_error = abs ( 1.0_wp - val )

        do j = 1 , i - 1

          val = my_ddot(nmo,trans_%u_irrep_block(block_sym)%val(:,i),1,&
              trans_%u_irrep_block(block_sym)%val(:,j),1)

          if ( abs( val ) > max_orthogonality_error ) max_orthogonality_error = abs ( val )
           
        end do      
 
      end do

      if ( log_print_ == 1 ) then
        if ( ( max_normalization_error > max_error_tolerance ) .or. &
           & ( max_orthogonality_error > max_error_tolerance ) ) then
          write(fid_,'(a,1x,i1,5x,a,1x,i3,5x,a,1x,es10.3,5x,a,1x,es10.3)')'irrep:',block_sym,'nmo:',block_dim,&
               & 'max(normalization_error):',max_normalization_error,'max(orthogonality_error):',max_orthogonality_error
        endif 
      endif

      ! ******************************
      ! deallocate tempporary matrices
      ! ******************************
     
      deallocate(tmp_mat_1,tmp_mat_2)
      deallocate(K2,d,X)

      compute_block_exponential = 0

      return
    end function compute_block_exponential

    integer function compute_compact_block_exponential(K,block_sym,max_dim)
      implicit none

      integer, intent(in)  :: block_sym,max_dim
      real(wp), intent(in) :: K(max_dim,max_dim)

      integer :: nmo,nfzc,internal_end,rotatable_internal_dim
      integer :: external_dim,external_rank,rank
      integer :: i,info,lwork
      real(wp) :: work_query(1)
      real(wp), allocatable :: external_basis(:,:),tau(:),work(:)
      real(wp), allocatable :: kv(:,:),projected_k(:,:),projected_u(:,:)
      real(wp), allocatable :: va(:,:)

      compute_compact_block_exponential = 1
      nmo                    = trans_%nmopi(block_sym)
      nfzc                   = nfzcpi_(block_sym)
      internal_end           = ndocpi_(block_sym) + nactpi_(block_sym)
      rotatable_internal_dim = internal_end - nfzc
      external_dim           = nmo - internal_end
      external_rank          = min(external_dim,rotatable_internal_dim)
      rank                   = rotatable_internal_dim + external_rank

      if ( nmo <= 0 .or. rotatable_internal_dim <= 0 .or. &
           external_dim <= 0 ) return
      if ( rank <= 0 .or. rank >= nmo ) return
      if ( size(trans_%compact_v(block_sym)%val,1) /= nmo ) return
      if ( size(trans_%compact_v(block_sym)%val,2) < rank ) return
      if ( size(trans_%compact_a(block_sym)%val,1) < rank ) return
      if ( size(trans_%compact_a(block_sym)%val,2) < rank ) return

      trans_%compact_v(block_sym)%val = 0.0_wp
      trans_%compact_a(block_sym)%val = 0.0_wp

      ! The nonfrozen internal coordinate vectors span the only internal block
      ! exponentiated by the legacy dense path.  A thin QR of
      ! K(external,rotatable-internal) supplies a basis for the remaining part
      ! of range(K).  Keeping all min(E,I) QR columns is deliberate: it makes
      ! the representation exact even for rank-deficient trial steps without
      ! a numerical-rank threshold.  Frozen occupied vectors remain in the
      ! identity complement.
      do i = 1 , rotatable_internal_dim
        trans_%compact_v(block_sym)%val(nfzc+i,i) = 1.0_wp
      end do

      if ( external_dim <= rotatable_internal_dim ) then
        do i = 1 , external_dim
          trans_%compact_v(block_sym)%val(internal_end+i, &
               rotatable_internal_dim+i) = 1.0_wp
        end do
      else
        allocate(external_basis(external_dim,external_rank))
        allocate(tau(external_rank))
        external_basis = K(internal_end+1:nmo,nfzc+1:internal_end)

        call dgeqrf(external_dim,external_rank,external_basis,external_dim, &
             tau,work_query,-1,info)
        if ( info /= 0 ) then
          deallocate(external_basis,tau)
          return
        end if
        lwork = max(1,int(work_query(1)))
        allocate(work(lwork))
        call dgeqrf(external_dim,external_rank,external_basis,external_dim, &
             tau,work,lwork,info)
        deallocate(work)
        if ( info /= 0 ) then
          deallocate(external_basis,tau)
          return
        end if

        call dorgqr(external_dim,external_rank,external_rank,external_basis, &
             external_dim,tau,work_query,-1,info)
        if ( info /= 0 ) then
          deallocate(external_basis,tau)
          return
        end if
        lwork = max(1,int(work_query(1)))
        allocate(work(lwork))
        call dorgqr(external_dim,external_rank,external_rank,external_basis, &
             external_dim,tau,work,lwork,info)
        deallocate(work,tau)
        if ( info /= 0 ) then
          deallocate(external_basis)
          return
        end if
        trans_%compact_v(block_sym)%val(internal_end+1:nmo, &
             rotatable_internal_dim+1:rank) = external_basis
        deallocate(external_basis)
      end if

      allocate(kv(nmo,rank))
      allocate(projected_k(rank,rank))
      allocate(projected_u(rank,rank))
      allocate(va(nmo,rank))

      call dgemm('N','N',nmo,rank,nmo,1.0_wp,K,max_dim, &
           trans_%compact_v(block_sym)%val,nmo,0.0_wp,kv,nmo)
      call dgemm('T','N',rank,rank,nmo,1.0_wp, &
           trans_%compact_v(block_sym)%val,nmo,kv,nmo,0.0_wp,projected_k,rank)

      info = skew_symmetric_exponential(projected_k,rank,projected_u)
      if ( info /= 0 ) then
        deallocate(kv,projected_k,projected_u,va)
        return
      end if

      trans_%compact_a(block_sym)%val(1:rank,1:rank) = projected_u
      do i = 1 , rank
        trans_%compact_a(block_sym)%val(i,i) = &
             trans_%compact_a(block_sym)%val(i,i) - 1.0_wp
      end do

      ! Reconstruct the full U for the unchanged host transforms:
      ! U = I + V * (exp(V^T K V)-I) * V^T.
      call dgemm('N','N',nmo,rank,rank,1.0_wp, &
           trans_%compact_v(block_sym)%val,nmo, &
           trans_%compact_a(block_sym)%val,rank,0.0_wp,va,nmo)
      trans_%u_irrep_block(block_sym)%val = 0.0_wp
      do i = 1 , nmo
        trans_%u_irrep_block(block_sym)%val(i,i) = 1.0_wp
      end do
      call dgemm('N','T',nmo,nmo,rank,1.0_wp,va,nmo, &
           trans_%compact_v(block_sym)%val,nmo,1.0_wp, &
           trans_%u_irrep_block(block_sym)%val,nmo)

      trans_%compact_rank(block_sym) = rank

      deallocate(kv,projected_k,projected_u,va)
      compute_compact_block_exponential = 0
      return
    end function compute_compact_block_exponential

    integer function skew_symmetric_exponential(K,n,U)
      implicit none

      integer, intent(in)  :: n
      real(wp), intent(in) :: K(n,n)
      real(wp), intent(out) :: U(n,n)

      integer :: i,info,lwork
      real(wp) :: value,work_query(1)
      real(wp), allocatable :: K2(:,:),X(:,:),tmp_1(:,:),tmp_2(:,:)
      real(wp), allocatable :: eigenvalues(:),work(:)

      skew_symmetric_exponential = 1
      allocate(K2(n,n),X(n,n),tmp_1(n,n),tmp_2(n,n))
      allocate(eigenvalues(n))

      call dgemm('N','N',n,n,n,1.0_wp,K,n,K,n,0.0_wp,K2,n)
      X = K2
      call dsyev('V','U',n,X,n,eigenvalues,work_query,-1,info)
      if ( info /= 0 ) then
        deallocate(K2,X,tmp_1,tmp_2,eigenvalues)
        return
      end if
      lwork = max(1,int(work_query(1)))
      allocate(work(lwork))
      call dsyev('V','U',n,X,n,eigenvalues,work,lwork,info)
      deallocate(work)
      if ( info /= 0 ) then
        deallocate(K2,X,tmp_1,tmp_2,eigenvalues)
        return
      end if

      do i = 1 , n
        eigenvalues(i) = sqrt(abs(eigenvalues(i)))
        value = 1.0_wp
        if ( eigenvalues(i) /= 0.0_wp ) &
             value = sin(eigenvalues(i)) / eigenvalues(i)
        tmp_1(:,i) = value * X(:,i)
      end do
      call dgemm('N','T',n,n,n,1.0_wp,X,n,tmp_1,n,0.0_wp,tmp_2,n)
      call dgemm('N','N',n,n,n,1.0_wp,K,n,tmp_2,n,0.0_wp,U,n)

      do i = 1 , n
        tmp_1(:,i) = cos(eigenvalues(i)) * X(:,i)
      end do
      call dgemm('N','T',n,n,n,1.0_wp,X,n,tmp_1,n,1.0_wp,U,n)

      deallocate(K2,X,tmp_1,tmp_2,eigenvalues)
      skew_symmetric_exponential = 0
      return
    end function skew_symmetric_exponential

    integer function gather_kappa_block(kappa_in,block,block_sym)
      implicit none
      real(wp) :: kappa_in(:)
      real(wp) :: block(:,:)
      integer, intent(in) :: block_sym
      integer :: i,j,n_ij,ij,i_irrep,j_irrep,i_class,j_class,j_class_start,j_start

      gather_kappa_block = 1

      ! number of orbital pairs in this block
      n_ij = trans_%npairpi(block_sym)

      if ( n_ij == 0 ) then
        gather_kappa_block = 0
        return
      end if

      ! figure out first/last orbital pair index for this block 

      ij = 0
      if ( block_sym > 1 ) ij = sum(trans_%npairpi(1:block_sym-1))
      
      ! initialize matrix block
      block = 0.0_wp

      ! loop over i/j pairs in this block

      ! the loop structure below cycles through the possible rotation pairs
      ! rotation pairs are sorted according to symmetry and for each irrep,
      ! the rotation pairs are sorted according to orbital classes: ad,ed,aa,ea
      ! for each pair j>i; for more details, see subroutine setup_rotation_indeces in focas_main.F90

 
      do i_class = 1 , 3

        j_class_start = i_class + 1

        if ( ( include_aa_rot_ == 1 ) .and. ( i_class == 2 ) ) j_class_start = i_class

        do j_class = j_class_start , 3

          do i = first_index_(block_sym,i_class) , last_index_(block_sym,i_class)

            j_start = first_index_(block_sym,j_class)

            if ( i_class == j_class ) j_start = i + 1

            do j = j_start , last_index_(block_sym,j_class)

              ! figure out symmetry_reduced indeces

              i_irrep                = trans_%class_to_irrep_map(i)
              j_irrep                = trans_%class_to_irrep_map(j)

              ! update gradient index

              ij                     = ij + 1

              ! copy into matrix

              block(j_irrep,i_irrep) =  kappa_in(ij)
              block(i_irrep,j_irrep) = -kappa_in(ij)

            end do

          end do

        end do    

      end do ! end ij loop

      gather_kappa_block = 0

    end function gather_kappa_block

end module focas_exponential
