#ifndef local_essential_tree_h
#define local_essential_tree_h
#include <map>
#include "alltoall.h"
#include "hilbert.h"
#include "timer.h"
#define SEND_ALL 0 //! Set to 1 for debugging

namespace exafmm_t {
  template <typename T> using BodyMap = std::multimap<uint64_t, Body<T>>;
  template <typename T> using NodeMap = std::map<uint64_t, Node<T>>;
  // int LEVEL;                                    //!< Octree level used for partitioning
  // std::vector<int> OFFSET;                      //!< Offset of Hilbert index for partitions

  //! Distance between cell center and edge of a remote domain
  template <typename T>
  real_t getDistance(Node<T>* C, int irank, std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t distance = R0;
    real_t R = R0 / (1 << LEVEL);
    for (int key=OFFSET[irank]; key<OFFSET[irank+1]; key++) {
      ivec3 iX = get3DIndex(key, LEVEL);
      vec3 X = getCoordinates(iX, LEVEL, X0, R0);
      vec3 Xmin = X - R;
      vec3 Xmax = X + R;
      vec3 dX;
			for (int d=0; d<3; d++) {
				dX[d] = (C->x[d] > Xmax[d]) * (C->x[d] - Xmax[d]) + (C->x[d] < Xmin[d]) * (C->x[d] - Xmin[d]);
			}
			distance = std::min(distance, norm(dX));
    }
    return distance;
  }

  //! Recursive call to pre-order tree traversal for selecting cells to send
  template <typename T>
  void selectCells(Node<T>* Cj, int irank, Bodies<T>& bodyBuffer, std::vector<int> & sendBodyCount,
                   Nodes<T>& cellBuffer, std::vector<int> & sendCellCount,
                   std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
    real_t R = getDistance(Cj, irank, OFFSET, LEVEL, X0, R0);
    real_t THETA = 0.5;
    real_t R2 = R * R * THETA * THETA;
    sendCellCount[irank]++;
    cellBuffer.push_back(*Cj);

    if (R2 <= (Cj->r + Cj->r) * (Cj->r + Cj->r)) {   // if near range
      if (Cj->is_leaf) {
        sendBodyCount[irank] += Cj->nsrcs;
        for (int b=0; b<Cj->nsrcs; b++) {
          bodyBuffer.push_back(Cj->first_src[b]);
        }
      } else {
        for (auto & child : Cj->children) {
          selectCells(child, irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                      OFFSET, LEVEL, X0, R0);
        }
      }
    }
  }

  template <typename T>
  void whatToSend(Nodes<T> & cells, Bodies<T> & bodyBuffer, std::vector<int> & sendBodyCount,
                  Nodes<T> & cellBuffer, std::vector<int> & sendCellCount,
                  std::vector<int>& OFFSET, int LEVEL, vec3 X0, real_t R0) {
#if SEND_ALL //! Send everything (for debugging)
    for (int irank=0; irank<MPISIZE; irank++) {
      sendCellCount[irank] = cells.size();
      for (size_t i=0; i<cells.size(); i++) {
        if (cells[i].is_leaf) {
          sendBodyCount[irank] += cells[i].nsrcs;
          for (int b=0; b<cells[i].nsrcs; b++) {
            bodyBuffer.push_back(cells[i].first_src[b]);
          }
        }
      }
      cellBuffer.insert(cellBuffer.end(), cells.begin(), cells.end());
    }
#else //! Send only necessary cells
    for (int irank=0; irank<MPISIZE; irank++) {
      selectCells(&cells[0], irank, bodyBuffer, sendBodyCount, cellBuffer, sendCellCount,
                  OFFSET, LEVEL, X0, R0);
    }
#endif
  }

  //! MPI communication for local essential tree
  template <typename T>
  void localEssentialTree(Bodies<T>& sources, Bodies<T>& targets, Nodes<T>& nodes,
                          NodePtrs<T>& leafs, NodePtrs<T>& nonleafs,
                          FmmBase<T>& fmm, std::vector<int>& OFFSET) {
    vec3 x0 = nodes[0].x;
    real_t r0 = nodes[0].r;
    int nsurf = nodes[0].up_equiv.size();
    std::vector<int> sendBodyCount(MPISIZE, 0);
    std::vector<int> recvBodyCount(MPISIZE, 0);
    std::vector<int> sendBodyDispl(MPISIZE, 0);
    std::vector<int> recvBodyDispl(MPISIZE, 0);
    std::vector<int> sendCellCount(MPISIZE, 0);
    std::vector<int> recvCellCount(MPISIZE, 0);
    std::vector<int> sendCellDispl(MPISIZE, 0);
    std::vector<int> recvCellDispl(MPISIZE, 0);
    Bodies<T> sendBodies, recvBodies;
    Nodes<T> sendCells, recvCells;
    //! Decide which nodes & bodies to send
    whatToSend(nodes, sendBodies, sendBodyCount, sendCells, sendCellCount,
               OFFSET, fmm.depth, x0, r0);
    //! Use alltoall to get recv count and calculate displacement (defined in alltoall.h)
    getCountAndDispl(sendBodyCount, sendBodyDispl, recvBodyCount, recvBodyDispl);
    getCountAndDispl(sendCellCount, sendCellDispl, recvCellCount, recvCellDispl);
    //! Alltoallv for nodes (defined in alltoall.h)
    alltoallCells(sendCells, sendCellCount, sendCellDispl, recvCells, recvCellCount, recvCellDispl);
    //! Alltoallv for sources (defined in alltoall.h)
    alltoallBodies(sendBodies, sendBodyCount, sendBodyDispl, recvBodies, recvBodyCount, recvBodyDispl);

    if (MPIRANK == 0) std::cout << "number of nodes" << std::endl;
    printMPI(nodes.size());
    if (MPIRANK == 0) std::cout << "number of sources" << std::endl;
    printMPI(sources.size());
    if (MPIRANK == 0) std::cout << "number of nodes received" << std::endl;
    for (int i=0; i<MPISIZE; i++)
      printMPI(recvCellCount[i]);
    if (MPIRANK == 0) std::cout << "number of sources received" << std::endl;
    for (int i=0; i<MPISIZE; i++)
      printMPI(recvBodyCount[i]);
  }
}
#endif
