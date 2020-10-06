#ifndef alltoall_h
#define alltoall_h
#include "exafmm_t.h"

namespace exafmm_t {
  //! Use alltoall to get recv count and calculate displacement from it
  void getCountAndDispl(std::vector<int> & sendCount, std::vector<int> & sendDispl,
                        std::vector<int> & recvCount, std::vector<int> & recvDispl) {
    MPI_Alltoall(&sendCount[0], 1, MPI_INT, &recvCount[0], 1, MPI_INT, MPI_COMM_WORLD);
    for (int irank=0; irank<MPISIZE-1; irank++) {
      sendDispl[irank+1] = sendDispl[irank] + sendCount[irank];
      recvDispl[irank+1] = recvDispl[irank] + recvCount[irank];
    }
  }

  //! Alltoallv for bodies
  template <typename T>
  void alltoallBodies(Bodies<T>& sendBodies, std::vector<int> & sendBodyCount, std::vector<int> & sendBodyDispl,
                      Bodies<T>& recvBodies, std::vector<int> & recvBodyCount, std::vector<int> & recvBodyDispl) {
    MPI_Datatype MPI_BODY;
    MPI_Type_contiguous(sizeof(sendBodies[0]), MPI_CHAR, &MPI_BODY);
    MPI_Type_commit(&MPI_BODY);
    recvBodies.resize(recvBodyDispl[MPISIZE-1]+recvBodyCount[MPISIZE-1]);
    MPI_Alltoallv(&sendBodies[0], &sendBodyCount[0], &sendBodyDispl[0], MPI_BODY,
                  &recvBodies[0], &recvBodyCount[0], &recvBodyDispl[0], MPI_BODY, MPI_COMM_WORLD);
  }
}
#endif
