SUBROUTINE cmds2(nelems, delta, coords)
  IMPLICIT NONE
  !
  INTEGER, INTENT(IN) :: nelems
  REAL*8, DIMENSION(nelems, nelems), INTENT(IN) :: delta
  REAL*8, DIMENSION(nelems, 2), INTENT(OUT) :: coords
  !
  INTEGER :: i
  REAL*8, DIMENSION(nelems, nelems) :: J
  REAL*8, DIMENSION(nelems, nelems) :: B
  REAL*8, DIMENSION(nelems) :: colsum, rowsum
  REAL*8 :: totalSum = 0.
  REAL*8, DIMENSION(nelems) :: W
  REAL*8, DIMENSION(nelems*4) :: WORK
  INTEGER, DIMENSION(1) :: l1, l2
  REAL*8 :: lambda1, lambda2
  INTEGER :: INFO

  J = 0.0
  do i=1,nelems; J(i,i)=1.0; end do
  J = J - 1.0/nelems
  
  B = -0.5 * MATMUL(MATMUL(J, delta), J)

  CALL DSYEV('V', 'U', nelems, B, nelems, W, WORK, nelems*4, INFO)

  W = ABS(W)
  l1 = MAXLOC(W) 
  lambda1 = W(l1(1)) 
  W(l1(1)) = 0.
  l2 = MAXLOC(W)
  lambda2 = W(l2(1)) 
  !WRITE(*,*) l1, l2
  coords(:, 1) = B(:, l1(1)) * SQRT(lambda1)
  coords(:, 2) = B(:, l2(1)) * SQRT(lambda2)
END SUBROUTINE
