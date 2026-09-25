#include "GlobalSpaceTimeFESpace.hpp"
#include <diffop_impl.hpp>

namespace ngcomp
 {
 
  
  GlobalCompoundSpaceTimeSpace :: GlobalCompoundSpaceTimeSpace (shared_ptr<MeshAccess> ama,
		     const Flags & flags, bool parseflags) 
	  : CompoundFESpace(ama, flags, parseflags)
  {  
    ;
  } 
 
  
  GlobalCompoundSpaceTimeSpace :: GlobalCompoundSpaceTimeSpace (shared_ptr<MeshAccess> ama,
		     const Array<shared_ptr<FESpace>> & aspaces,
		     const Flags & flags, bool parseflags) 
    : CompoundFESpace(ama, aspaces, flags, parseflags) 
  {
    ;
  }



 void ExportGlobalCompoundSpaceTimeSpace (py::module m)
 {

  typedef FESpace FES;
  
  
  m.def("GlobalCompoundSpaceTimeSpace", [] (
                                        shared_ptr<FES> basefes,
					int N, 
                                        py::object dirichlet,
					py::kwargs kwargs
                                        )
  {


    std::cout << "Hello from GlobalCompoundSpaceTimeSpace " << std::endl;
    //shared_ptr<SpaceTimeFESpace> ret = nullptr;
    //Flags flags = py::extract<Flags> (bpflags)();
    //Flags flags = bp::extract<Flags> (bpflags)();
    //auto spaces = makeCArrayUnpackWrapper<PyWrapper<FESpace>> (lspaces);
    
    auto flags = CreateFlagsFromKwArgs(kwargs);
    shared_ptr<MeshAccess> ma = basefes->GetMeshAccess();
    
    Array<shared_ptr<FESpace>> spaces(N); 
    for (int i = 0; i < N; i++)
      spaces[i] = basefes;

    
    if (py::isinstance<py::list>(dirichlet)) {
        flags.SetFlag("dirichlet", makeCArray<double>(py::list(dirichlet)));

    }

    if (py::isinstance<py::str>(dirichlet))
    {
        Array<double> dirlist;
        Region dir(ma, BND, dirichlet.cast<string>());
        for (int i = 0; i < ma->GetNBoundaries(); i++)
            if (dir.Mask()[i])
              dirlist.Append (i+1);
        flags.SetFlag("dirichlet", dirlist);
    }


    //auto tfe = dynamic_pointer_cast<ScalarFiniteElement<1>>(fe);
    //cout << tfe << endl;
    //if(tfe == nullptr)
    //  cout << IM(1) << "Warning! tfe == nullptr" << endl;

    //ret = make_shared<SpaceTimeFESpace> (ma, basefes,tfe, flags);
    
    //auto fes = GlobalCompoundSpaceTimeSpace(spaces[0]->GetMeshAccess(), spaces, flags);  
    //TestClass tmp;

    shared_ptr<FESpace> fes = make_shared<GlobalCompoundSpaceTimeSpace> (spaces[0]->GetMeshAccess(), spaces, flags);


    //LocalHeap lh (heapsize, "SpaceTimeFESpace::Update-heap", true);
    //ret->Update();
    //ret->FinalizeUpdate();
    return fes;
  },
       py::arg("fes"),
       py::arg("N"),
       py::arg("dirichlet")=py::none(),
       docu_string(R"raw_string(lalalala
  )raw_string")
   );
  

 }



 }
