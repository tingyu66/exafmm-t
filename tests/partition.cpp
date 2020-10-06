#include <numeric>    // std::accumulate
#include "dataset.h"
#include "exafmm_t.h"
#include "partition.h"

using namespace exafmm_t;

int main(int argc, char **argv) {
  Args args(argc, argv);
  startMPI(argc, argv);

  int n = args.numBodies;
  Bodies<real_t> sources = init_sources<real_t>(n, args.distribution, MPIRANK);
  Bodies<real_t> targets = init_targets<real_t>(n, args.distribution, MPIRANK+10);
  
  vec3 x0;
  real_t r0;
  allreduceBounds(sources, targets, x0, r0);
  std::vector<int> offset;
  partition(sources, targets, x0, r0, offset, args.maxlevel);

  // print partition information
  size_t nsrcs = sources.size();
  size_t ntrgs = targets.size();
  int size = sizeof(nsrcs);
  std::vector<size_t> nsrcs_recv(MPISIZE);
  std::vector<size_t> ntrgs_recv(MPISIZE);
  MPI_Gather(&nsrcs, size, MPI_BYTE, &nsrcs_recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Gather(&ntrgs, size, MPI_BYTE, &ntrgs_recv[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);

  if (MPIRANK == 0) {
    std::cout << "number of sources: " << std::endl;
    for (int irank=0; irank<MPISIZE; irank++)
      std::cout << nsrcs_recv[irank] << " ";
    std::cout << std::endl;

    std::cout << "number of targets: " << std::endl;
    for (int irank=0; irank<MPISIZE; irank++)
      std::cout << ntrgs_recv[irank] << " ";
    std::cout << std::endl;

    size_t nsrcs_total = std::accumulate(nsrcs_recv.begin(), nsrcs_recv.end(), 0);
    size_t ntrgs_total = std::accumulate(ntrgs_recv.begin(), ntrgs_recv.end(), 0);
    assert(nsrcs_total == n * MPISIZE);
    assert(ntrgs_total == n * MPISIZE);
    std::cout << "assertion passed!" << std::endl;

    std::cout << "r0: " << r0 << std::endl;
    std::cout << "x0: " << x0 << std::endl;
  }

  stopMPI();
}
