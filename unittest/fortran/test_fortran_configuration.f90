FUNCTION f_lammps_version() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_version

  f_lammps_version = lmp%version()
END FUNCTION f_lammps_version

FUNCTION f_lammps_os_info(ptr) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_ptr, c_char, c_int, c_size_t, &
    C_F_POINTER
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp, c_strlen
  IMPLICIT NONE
  TYPE(c_ptr), INTENT(IN), VALUE :: ptr
  INTEGER(c_int) :: f_lammps_os_info
  CHARACTER(LEN=:), ALLOCATABLE :: string, os_info
  CHARACTER(LEN=1, KIND=c_char), POINTER :: C_string(:)
  INTEGER(c_size_t) :: length, i

  length = c_strlen(ptr)
  CALL C_F_POINTER(ptr, C_string, [length])
  ALLOCATE(CHARACTER(LEN=length) :: string)
  DO i = 1, length
    string(i:i) = C_string(i)
  END DO

  ALLOCATE(CHARACTER(LEN=1000) :: os_info)
  CALL lmp%get_os_info(os_info)
  os_info = TRIM(os_info)

  IF (os_info(1:length) == string) THEN
    f_lammps_os_info = 1_c_int
  ELSE
    f_lammps_os_info = 0_c_int
  END IF
END FUNCTION f_lammps_os_info

FUNCTION f_lammps_mpi_support() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_mpi_support

  IF (lmp%config_has_mpi_support()) THEN
    f_lammps_mpi_support = 1_c_int
  ELSE
    f_lammps_mpi_support = 0_c_int
  END IF
END FUNCTION f_lammps_mpi_support

FUNCTION f_lammps_gzip_support() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_gzip_support

  IF (lmp%config_has_gzip_support()) THEN
    f_lammps_gzip_support = 1_c_int
  ELSE
    f_lammps_gzip_support = 0_c_int
  END IF
END FUNCTION f_lammps_gzip_support

FUNCTION f_lammps_png_support() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_png_support

  IF (lmp%config_has_png_support()) THEN
    f_lammps_png_support = 1_c_int
  ELSE
    f_lammps_png_support = 0_c_int
  END IF
END FUNCTION f_lammps_png_support

FUNCTION f_lammps_jpeg_support() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_jpeg_support

  IF (lmp%config_has_jpeg_support()) THEN
    f_lammps_jpeg_support = 1_c_int
  ELSE
    f_lammps_jpeg_support = 0_c_int
  END IF
END FUNCTION f_lammps_jpeg_support

FUNCTION f_lammps_ffmpeg_support() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_ffmpeg_support

  IF (lmp%config_has_ffmpeg_support()) THEN
    f_lammps_ffmpeg_support = 1_c_int
  ELSE
    f_lammps_ffmpeg_support = 0_c_int
  END IF
END FUNCTION f_lammps_ffmpeg_support

FUNCTION f_lammps_has_exceptions() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_has_exceptions

  IF (lmp%config_has_exceptions()) THEN
    f_lammps_has_exceptions = 1_c_int
  ELSE
    f_lammps_has_exceptions = 0_c_int
  END IF
END FUNCTION f_lammps_has_exceptions

FUNCTION f_lammps_has_package(Cname) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int, c_size_t, c_ptr, c_char, &
    C_F_POINTER
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp, c_strlen
  IMPLICIT NONE
  TYPE(c_ptr), VALUE :: Cname
  INTEGER(c_int) :: f_lammps_has_package
  CHARACTER(LEN=1,KIND=c_char), DIMENSION(:), POINTER :: Fname
  CHARACTER(LEN=:), ALLOCATABLE :: name
  INTEGER(c_size_t) :: length
  INTEGER :: i

  length = c_strlen(Cname)
  CALL C_F_POINTER(Cname, Fname, [length])
  ALLOCATE(CHARACTER(LEN=length) :: name)
  DO i = 1, length
    name(i:i) = Fname(i)
  END DO
  IF (lmp%config_has_package(name)) THEN
    f_lammps_has_package = 1_c_int
  ELSE
    f_lammps_has_package = 0_c_int
  END IF
END FUNCTION f_lammps_has_package

FUNCTION f_lammps_package_count() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_package_count

  f_lammps_package_count = lmp%config_package_count()
END FUNCTION f_lammps_package_count

FUNCTION f_lammps_package_name(idx) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_ptr, c_null_ptr, c_int
  USE keepstuff, ONLY : lmp, f2c_string
  IMPLICIT NONE
  INTEGER(c_int), VALUE :: idx
  TYPE(c_ptr) :: f_lammps_package_name
  CHARACTER(LEN=80) :: buffer

  CALL lmp%config_package_name(idx, buffer)
  IF (LEN_TRIM(buffer) > 0) THEN
    f_lammps_package_name = f2c_string(buffer)
  ELSE
    f_lammps_package_name = c_null_ptr
  END IF
END FUNCTION f_lammps_package_name

FUNCTION f_lammps_installed_packages(idx) BIND(C) RESULT(package)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int, c_ptr, c_null_ptr
  USE keepstuff, ONLY : lmp, f2c_string
  IMPLICIT NONE
  INTEGER(c_int), INTENT(IN), VALUE :: idx
  TYPE(c_ptr) :: package
  CHARACTER(LEN=:), DIMENSION(:), ALLOCATABLE :: all_packages

  CALL lmp%installed_packages(all_packages)

  IF (idx > SIZE(all_packages) .OR. idx <= 0) THEN
    package = c_null_ptr
  ELSE
    package = f2c_string(all_packages(idx))
  END IF
END FUNCTION f_lammps_installed_packages

FUNCTION f_lammps_config_accelerator(package, category, setting) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int, c_ptr, c_size_t, c_char, &
      C_F_POINTER
  USE keepstuff, ONLY : lmp, c_strlen
  USE LIBLAMMPS
  IMPLICIT NONE
  TYPE(c_ptr), VALUE :: package, category, setting
  INTEGER(c_int) :: f_lammps_config_accelerator
  INTEGER(c_size_t) :: len_package, len_category, len_setting
  CHARACTER(LEN=1,KIND=c_char), POINTER :: Cpackage(:),Ccategory(:),Csetting(:)
  CHARACTER(LEN=:), ALLOCATABLE :: Fpackage, Fcategory, Fsetting
  INTEGER :: i
  LOGICAL :: configured

  len_package = c_strlen(package)
  len_category = c_strlen(category)
  len_setting = c_strlen(setting)
  ALLOCATE(CHARACTER(LEN=len_package) :: Fpackage)
  ALLOCATE(CHARACTER(LEN=len_category) :: Fcategory)
  ALLOCATE(CHARACTER(LEN=len_setting) :: Fsetting)
  CALL C_F_POINTER(package, Cpackage, [len_package])
  CALL C_F_POINTER(category, Ccategory, [len_category])
  CALL C_F_POINTER(setting, Csetting, [len_setting])
  DO i = 1, len_package
    Fpackage(i:i) = Cpackage(i)
  END DO
  DO i = 1, len_category
    Fcategory(i:i) = Ccategory(i)
  END DO
  DO i = 1, len_setting
    Fsetting(i:i) = Csetting(i)
  END DO

  configured = lmp%config_accelerator(Fpackage, Fcategory, Fsetting)

  IF (configured) THEN
    f_lammps_config_accelerator = 1_c_int
  ELSE
    f_lammps_config_accelerator = 0_c_int
  END IF

END FUNCTION f_lammps_config_accelerator

FUNCTION f_lammps_has_gpu() BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_int
  USE keepstuff, ONLY : lmp
  IMPLICIT NONE
  INTEGER(c_int) :: f_lammps_has_gpu

  IF (lmp%has_gpu_device()) THEN
    f_lammps_has_gpu = 1_c_int
  ELSE
    f_lammps_has_gpu = 0_c_int
  END IF
END FUNCTION f_lammps_has_gpu

FUNCTION f_lammps_get_gpu_info(buf_size) RESULT(info) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_size_t, c_ptr
  USE keepstuff, ONLY : lmp, f2c_string
  IMPLICIT NONE
  INTEGER(c_size_t), VALUE :: buf_size
  TYPE(c_ptr) :: info
  CHARACTER(LEN=buf_size) :: string

  CALL lmp%get_gpu_device_info(string)
  info = f2c_string(string)
END FUNCTION f_lammps_get_gpu_info

FUNCTION f_lammps_has_style(Ccategory, Cname) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_size_t, c_ptr, c_char, c_int, &
    C_F_POINTER
  USE keepstuff, ONLY : lmp, c_strlen
  IMPLICIT NONE
  TYPE(c_ptr), INTENT(IN), VALUE :: Ccategory, Cname
  INTEGER(c_int) :: f_lammps_has_style
  CHARACTER(LEN=1,KIND=c_char), DIMENSION(:), POINTER :: Fcategory, Fname
  INTEGER(c_size_t) :: category_len, name_len, i
  CHARACTER(LEN=:), ALLOCATABLE :: category, name

  category_len = c_strlen(Ccategory)
  name_len = c_strlen(Cname)
  CALL C_F_POINTER(Ccategory, Fcategory, [category_len])
  CALL C_F_POINTER(Cname, Fname, [name_len])
  ALLOCATE(CHARACTER(LEN=category_len) :: category)
  ALLOCATE(CHARACTER(LEN=name_len) :: name)
  DO i = 1, category_len
    category(i:i) = Fcategory(i)
  END DO
  DO i = 1, name_len
    name(i:i) = Fname(i)
  END DO
  IF (lmp%has_style(category, name)) THEN
    f_lammps_has_style = 1_c_int
  ELSE
    f_lammps_has_style = 0_c_int
  END IF
END FUNCTION f_lammps_has_style

FUNCTION f_lammps_style_count(ptr) BIND(C)
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : c_ptr, c_int, c_char, c_size_t, &
    C_F_POINTER
  USE LIBLAMMPS
  USE keepstuff, ONLY : lmp, c_strlen
  IMPLICIT NONE
  TYPE(c_ptr), VALUE :: ptr
  INTEGER(c_int) :: f_lammps_style_count
  CHARACTER(LEN=1, KIND=c_char), DIMENSION(:), POINTER :: C_category
  INTEGER(c_size_t) :: length, i
  CHARACTER(LEN=:), ALLOCATABLE :: category

  length = c_strlen(ptr)
  CALL C_F_POINTER(ptr, C_category, [length])
  ALLOCATE(CHARACTER(LEN=length) :: category)
  DO i = 1, length
    category(i:i) = C_category(i)
  END DO
  f_lammps_style_count = lmp%style_count(category)
END FUNCTION f_lammps_style_count
