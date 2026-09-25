project(invtools)

cmake_minimum_required(VERSION 3.5)

find_package(NGSolve CONFIG REQUIRED
  HINTS /usr $ENV{NETGENDIR}/.. /tmp/inst/usr /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)


add_ngsolve_python_module(invtools invtools.cpp
  comp/GlobalSpaceTimeFESpace.cpp 
  )

# check if CMAKE_INSTALL_PREFIX is set by user, otherwise install to user-local python modules directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.USER_SITE)" OUTPUT_VARIABLE install_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CMAKE_INSTALL_PREFIX ${install_dir} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

message("The python package will be installed to: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS invtools DESTINATION .)
