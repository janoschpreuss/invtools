#ifndef FILE_GLOBALSPACETIMEFESPACE
#define FILE_GLOBALSPACETIMEFESPACE

// Partially based on ngsolve/comp/numberfespace.hpp


#include <core/register_archive.hpp>

#include <finiteelement.hpp>
#include <diffop.hpp>
#include <symbolicintegrator.hpp>
#include <fespace.hpp>

#include <basevector.hpp>
#include <basematrix.hpp>
#include <meshaccess.hpp>
#include "ngsobject.hpp"
#include <python_comp.hpp>

namespace ngcomp
{
  using namespace ngla;

//class NGS_DLL_HEADER GlobalCompoundSpaceTimeSpace : public CompoundFESpace 
//

class TestClass {       
  public:             
    int someint;          
};


class GlobalCompoundSpaceTimeSpace : public CompoundFESpace 
  {
   
  ;

  public:
    /*
      constructor. 
      Arguments are the access to the mesh data structure,
      and the flags from the define command in the pde-file
      or the kwargs in the Python constructor.
    */
    ;
   
    GlobalCompoundSpaceTimeSpace (shared_ptr<MeshAccess> ama,
		     const Flags & flags, bool parseflags = false);

    GlobalCompoundSpaceTimeSpace (shared_ptr<MeshAccess> ama,
		     const Array<shared_ptr<FESpace>> & aspaces,
		     const Flags & flags, bool parseflags = false);


    // a name for our new fe-space
    string GetClassName () const override { return "GlobalCompoundSpaceTimeSpace"; }

    static DocInfo GetDocu();

    // void Update() override;
    
    //void GetDofNrs (ElementId ei, Array<DofId> & dnums) const override;
    //FiniteElement & GetFE (ElementId ei, Allocator & alloc) const override;

  };

  void ExportGlobalCompoundSpaceTimeSpace (py::module m);

}
#endif
