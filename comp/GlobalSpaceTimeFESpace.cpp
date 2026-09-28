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
		     const int andofspace,
		     const Flags & flags, bool parseflags) 
    : CompoundFESpace(ama, aspaces, flags, parseflags), ndofspace(andofspace)  
  {
    ;
  }


  void GlobalCompoundSpaceTimeSpace :: Update()
  { 
    std::cout << "Hello from CompoundFESpace :: Update" << std::endl;
    std::cout << "ndofspace = " << ndofspace << std::endl;
    FESpace::Update();
    if (low_order_space)
      low_order_space->Update();
    
    cummulative_nd.SetSize (spaces.Size()+1);
    cummulative_nd[0] = 0;
    for (int i = 0; i < spaces.Size(); i++)
      {
        if(do_subspace_update)
          spaces[i] -> Update();
	//cummulative_nd[i+1] = cummulative_nd[i] + spaces[i]->GetNDof();
	// shift dofs
	if (i <  spaces.Size() - 1 ) 
	 {
	   cummulative_nd[i+1] = cummulative_nd[i] + spaces[i]->GetNDof() - ndofspace;
	 }
        else 
	 {
	   cummulative_nd[i+1] = cummulative_nd[i] + spaces[i]->GetNDof();
	 }	 
      }

    SetNDof (cummulative_nd.Last()); 
    
    bool has_atomic = false;
    for (auto & space : spaces)
      if (space->HasAtomicDofs())
        has_atomic = true;
    if (has_atomic)
      {
        is_atomic_dof = BitArray(GetNDof());
        is_atomic_dof = false;
        for (int i = 0; i < spaces.Size(); i++)
          {
            FESpace & spacei = *spaces[i];
            IntRange r(cummulative_nd[i], cummulative_nd[i+1]);
            if (spacei.HasAtomicDofs())
              {
                for (size_t j = 0; j < r.Size(); j++)
                  if (spacei.IsAtomicDof(j))
                    is_atomic_dof.SetBit(r.begin()+j);
              }
          }       
      }
    // cout << "AtomicDofs = " << endl << is_atomic_dof << endl;

    // prol -> Update(*this);  // do we need that ? 
    
    std::cout << "GetNDof = " << this->GetNDof() << std::endl;
    UpdateCouplingDofArray();


    if (low_order_space)
      {
        shared_ptr<BaseMatrix> sum_emb;
        for (size_t i = 0; i < spaces.Size(); i++)
          {
            auto emb_i = spaces[i]->LowOrderEmbedding();
            auto hi_range = GetRange(i);
            auto lo_range = dynamic_pointer_cast<CompoundFESpace>(low_order_space)->GetRange(i);
            emb_i = make_shared<EmbeddedMatrix> (GetNDof(), hi_range, emb_i);
            emb_i = make_shared<EmbeddedTransposeMatrix> (low_order_space->GetNDof(), lo_range, emb_i);
            if (sum_emb)
              sum_emb = make_shared<SumMatrix> (sum_emb, emb_i);
            else
              sum_emb = emb_i;
          }
        low_order_embedding = sum_emb;
        // cout << "embedding = " << *low_order_embedding << endl;
      }

    
    if (print)
      {
	(*testout) << "Update compound fespace" << endl;
	(*testout) << "cumulative dofs start at " << cummulative_nd << endl;
      }
  }







 void ExportGlobalCompoundSpaceTimeSpace (py::module m)
 {

  typedef FESpace FES;
  
  
  m.def("GlobalCompoundSpaceTimeSpace", [] (
                                        shared_ptr<FES> basefes,
					int N,
				        int ndofspace,	
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

    shared_ptr<FESpace> fes = make_shared<GlobalCompoundSpaceTimeSpace> (spaces[0]->GetMeshAccess(), spaces, ndofspace, flags);


    //LocalHeap lh (heapsize, "SpaceTimeFESpace::Update-heap", true);
    fes->Update();
    fes->FinalizeUpdate();
    return fes;
  },
       py::arg("fes"),
       py::arg("N"),
       py::arg("ndofspace"),
       py::arg("dirichlet")=py::none(),
       docu_string(R"raw_string(lalalala
  )raw_string")
   );
  

 }



 }
