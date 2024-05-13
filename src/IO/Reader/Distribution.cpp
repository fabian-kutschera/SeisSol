#include "Distribution.hpp"

#include <IO/Datatype/Inference.hpp>
#include <IO/Datatype/MPIType.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/logger.h"

namespace {
static int getRank(std::size_t id, std::size_t count, int commsize) {
  const std::size_t piece1 = (count + commsize - 1) / commsize;
  const std::size_t rest = count % commsize;
  const std::size_t part1 = piece1 * rest;
  if (id < part1) {
    return id / piece1;
  } else {
    const std::size_t piece2 = count / commsize;
    return (id - part1) / piece2 + rest;
  }
}

template <typename T>
static std::vector<T> computeHistogram(const std::vector<T>& input) {
  std::vector<T> output(input.size() + 1);
  for (std::size_t i = 1; i < output.size(); ++i) {
    output[i] = output[i - 1] + input[i - 1];
  }
  return output;
}

template <typename T>
static std::vector<std::pair<T, int>> distributeIds(const std::vector<std::pair<T, int>>& source,
                                                    MPI_Comm comm,
                                                    MPI_Datatype sizetype,
                                                    MPI_Datatype datatype,
                                                    int tag) {
  int commsize;
  MPI_Comm_size(comm, &commsize);

  std::vector<std::size_t> sourceToSend(commsize);
  for (const auto& [_, j] : source) {
    ++sourceToSend[j];
  }
  std::vector<std::size_t> sourceToRecv(commsize);

  MPI_Alltoall(sourceToSend.data(), 1, sizetype, sourceToRecv.data(), 1, sizetype, comm);

  auto sourceToSendHistogram = computeHistogram(sourceToSend);

  std::vector<T> sortedSource(source.size());
  {
    auto tempHistogram = sourceToSendHistogram;
    for (const auto& [i, j] : source) {
      auto& position = tempHistogram[j];
      sortedSource[position] = i;
      ++position;
    }
  }

  auto sourceToRecvHistogram = computeHistogram(sourceToRecv);

  std::vector<T> intermediate(sourceToRecvHistogram.back());

  {
    std::vector<MPI_Request> requests(static_cast<std::size_t>(commsize) * 2, MPI_REQUEST_NULL);

    for (int i = 0; i < commsize; ++i) {
      if (sourceToSend[i] > 0) {
        MPI_Isend(sortedSource.data() + sourceToSendHistogram[i],
                  sourceToSend[i],
                  datatype,
                  i,
                  tag,
                  comm,
                  &requests[i]);
      }
      if (sourceToRecv[i] > 0) {
        MPI_Irecv(intermediate.data() + sourceToRecvHistogram[i],
                  sourceToRecv[i],
                  datatype,
                  i,
                  tag,
                  comm,
                  &requests[static_cast<std::size_t>(commsize) + i]);
      }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  std::vector<std::pair<T, int>> outputIntermediate(intermediate.size());
  int rank = 0;
  for (std::size_t i = 0; i < intermediate.size(); ++i) {
    if (sourceToRecvHistogram[rank + 1] <= i) {
      ++rank;
    }
    outputIntermediate[i] = std::pair<T, int>(intermediate[i], rank);
  }

  return outputIntermediate;
}

static std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
    matchRanks2(const std::vector<std::pair<std::size_t, int>>& intermediateSource,
                const std::vector<std::pair<std::size_t, int>>& intermediateTarget,
                const std::vector<std::size_t>& source,
                MPI_Comm comm,
                MPI_Datatype sizetype,
                MPI_Datatype datatype,
                int tag) {
  int commsize;
  MPI_Comm_size(comm, &commsize);

  std::vector<std::size_t> sendOffsets, sendReorder;

  std::vector<std::pair<std::pair<std::size_t, int>, int>> sourceToTargetRankMap;
  sourceToTargetRankMap.reserve(std::max(intermediateTarget.size(), intermediateSource.size()));
  {
    // for multiple source locations, we assume that all contain the same data, thus take one of
    // them for multiple target locations, we need to write to them all
    std::unordered_multimap<std::size_t, int> intermediateSourceMap;
    for (const auto& i : intermediateSource) {
      intermediateSourceMap.insert(i);
    }
    for (std::size_t i = 0; i < intermediateTarget.size(); ++i) {
      for (auto it = intermediateSourceMap.find(intermediateTarget[i].first);
           it != intermediateSourceMap.end() && it->first == intermediateTarget[i].first;
           ++it) {
        sourceToTargetRankMap.push_back({intermediateTarget[i], it->second});
      }
    }
  }

  auto sourceToTarget = distributeIds<std::pair<std::size_t, int>>(
      sourceToTargetRankMap, comm, sizetype, datatype, tag);
  {
    std::unordered_map<std::size_t, std::size_t> reorderMap;
    {
      std::vector<std::size_t> sourceCounter(commsize);
      for (std::size_t i = 0; i < sourceToTarget.size(); ++i) {
        ++sourceCounter[sourceToTarget[i].first.second];
      }
      sendOffsets = computeHistogram(sourceCounter);
    }
    {
      std::vector<std::size_t> tempHistogram(sendOffsets.begin(), sendOffsets.end());
      for (std::size_t i = 0; i < sourceToTarget.size(); ++i) {
        auto& position = tempHistogram[sourceToTarget[i].first.second];
        reorderMap[sourceToTarget[i].first.first] = position;
        ++position;
      }
    }

    sendReorder.resize(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      sendReorder[i] = reorderMap.at(source[i]);
    }
  }

  return {sendOffsets, sendReorder};
}

static std::pair<std::vector<std::size_t>, std::vector<std::size_t>> matchRanks(
    const std::vector<std::pair<std::pair<std::size_t, int>, int>>& sourceToTargetRankMap,
    const std::vector<std::size_t>& source,
    MPI_Comm comm,
    MPI_Datatype sizetype,
    MPI_Datatype datatype,
    int tag) {
  int commsize;
  MPI_Comm_size(comm, &commsize);

  std::vector<std::size_t> sendOffsets, sendReorder;

  auto sourceToTarget = distributeIds<std::pair<std::size_t, int>>(
      sourceToTargetRankMap, comm, sizetype, datatype, tag);
  {
    std::unordered_multimap<std::size_t, std::size_t> reorderMap;
    {
      std::vector<std::size_t> sourceCounter(commsize);
      for (std::size_t i = 0; i < sourceToTarget.size(); ++i) {
        ++sourceCounter[sourceToTarget[i].first.second];
      }
      sendOffsets = computeHistogram(sourceCounter);
    }
    {
      std::vector<std::size_t> tempHistogram(sendOffsets.begin(), sendOffsets.end());
      for (std::size_t i = 0; i < sourceToTarget.size(); ++i) {
        auto& position = tempHistogram[sourceToTarget[i].first.second];
        reorderMap.insert({sourceToTarget[i].first.first, position});
        ++position;
      }
    }

    sendReorder.reserve(source.size());
    for (std::size_t i = 0; i < source.size(); ++i) {
      for (auto it = reorderMap.find(source[i]); it != reorderMap.end() && it->first == source[i];
           ++it) {
        sendReorder.push_back(it->second);
      }
    }
  }

  return {sendOffsets, sendReorder};
}

} // namespace

namespace seissol::io::reader {
Distributor::Distributor(MPI_Comm comm) : comm(comm) {}

void Distributor::setup(const std::vector<std::size_t>& sourceIds,
                        const std::vector<std::size_t>& targetIds) {
  int commsize;
  int commrank;

  constexpr int tagToIntermediateSource = 20;
  constexpr int tagToIntermediateTarget = 21;
  constexpr int tagFromIntermediateSource = 22;
  constexpr int tagFromIntermediateTarget = 23;

  MPI_Comm_size(comm, &commsize);
  MPI_Comm_rank(comm, &commrank);

  MPI_Datatype sizetype =
      seissol::io::datatype::convertToMPI(seissol::io::datatype::inferDatatype<std::size_t>());
  MPI_Datatype pairtype = seissol::io::datatype::convertToMPI(
      seissol::io::datatype::inferDatatype<std::pair<std::size_t, int>>());

  auto sourceMax = *std::max_element(sourceIds.begin(), sourceIds.end());
  auto targetMax = *std::max_element(targetIds.begin(), targetIds.end());
  std::size_t localMax = std::max(sourceMax, targetMax);
  std::size_t globalMax;
  MPI_Allreduce(&localMax, &globalMax, 1, sizetype, MPI_MAX, comm);
  auto globalCount = globalMax + 1;

  // note that the following operations all need to be stable (i.e. order-preserving) for these
  // methods to work correctly right now

  std::vector<std::pair<std::size_t, int>> source;
  std::vector<std::pair<std::size_t, int>> target;

  source.reserve(sourceIds.size());
  target.reserve(targetIds.size());

  {
    std::unordered_set<std::size_t> seenIds;
    for (std::size_t i = 0; i < sourceIds.size(); ++i) {
      if (seenIds.find(sourceIds[i]) == seenIds.end()) {
        seenIds.insert(sourceIds[i]);
        source.push_back({sourceIds[i], getRank(sourceIds[i], globalCount, commsize)});
      }
    }
  }
  {
    std::unordered_set<std::size_t> seenIds;
    for (std::size_t i = 0; i < targetIds.size(); ++i) {
      if (seenIds.find(targetIds[i]) == seenIds.end()) {
        seenIds.insert(targetIds[i]);
        target.push_back({targetIds[i], getRank(targetIds[i], globalCount, commsize)});
      }
    }
  }

  auto intermediateSource =
      distributeIds<std::size_t>(source, comm, sizetype, sizetype, tagToIntermediateSource);
  auto intermediateTarget =
      distributeIds<std::size_t>(target, comm, sizetype, sizetype, tagToIntermediateTarget);

  // we need the same ordering, to transfer the IDs in the right order
  // TODO(David): may be removable
  std::sort(intermediateSource.begin(), intermediateSource.end());
  std::sort(intermediateTarget.begin(), intermediateTarget.end());

  std::vector<std::pair<std::pair<std::size_t, int>, int>> sourceToTargetRankMap;
  std::vector<std::pair<std::pair<std::size_t, int>, int>> targetToSourceRankMap;

  sourceToTargetRankMap.resize(intermediateTarget.size());
  targetToSourceRankMap.resize(intermediateTarget.size());

  {
    std::size_t iSource = 0;
    for (std::size_t i = 0; i < intermediateTarget.size(); ++i) {
      while (intermediateSource[iSource].first != intermediateTarget[i].first) {
        ++iSource;
      }
      targetToSourceRankMap[i] = {intermediateSource[iSource], intermediateTarget[i].second};
      sourceToTargetRankMap[i] = {intermediateTarget[i], intermediateSource[iSource].second};
    }
  }

  auto sendResult = matchRanks(
      sourceToTargetRankMap, sourceIds, comm, sizetype, pairtype, tagFromIntermediateSource);
  sendOffsets = sendResult.first;
  sendReorder = sendResult.second;

  auto recvResult = matchRanks(
      targetToSourceRankMap, targetIds, comm, sizetype, pairtype, tagFromIntermediateTarget);
  recvOffsets = recvResult.first;
  recvReorder = recvResult.second;

  // selftest (maybe only activate for debug?)
  {
    std::vector<std::size_t> targetCompare(targetIds.size());
    distribute(targetCompare.data(), sourceIds.data());
    for (std::size_t i = 0; i < targetIds.size(); ++i) {
      assert(targetIds[i] == targetCompare[i]);
      if (targetIds[i] != targetCompare[i]) {
        logError() << "Global ID redistribution mismatch. Checkpoint loading failed.";
      }
    }
  }
}

void Distributor::distributeInternal(void* target, const void* source, MPI_Datatype datatype) {
  constexpr int tag = 30;

  int commsize;
  int commrank;

  MPI_Comm_size(comm, &commsize);
  MPI_Comm_rank(comm, &commrank);

  int typesizeInt;
  MPI_Type_size(datatype, &typesizeInt);
  const std::size_t typesize = typesizeInt;

  std::vector<MPI_Request> requests(static_cast<std::size_t>(commsize) * 2, MPI_REQUEST_NULL);

  const char* sourceChar = reinterpret_cast<const char*>(source);
  char* targetChar = reinterpret_cast<char*>(target);

  char* sourceReordered = reinterpret_cast<char*>(std::malloc(typesize * sendReorder.size()));
  char* targetReordered = reinterpret_cast<char*>(std::malloc(typesize * recvReorder.size()));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < sendReorder.size(); ++i) {
    std::memcpy(sourceReordered + sendReorder[i] * typesize, sourceChar + i * typesize, typesize);
  }

  for (int i = 0; i < commsize; ++i) {
    if (sendOffsets[i + 1] > sendOffsets[i]) {
      MPI_Isend(sourceReordered + sendOffsets[i] * typesize,
                sendOffsets[i + 1] - sendOffsets[i],
                datatype,
                i,
                tag,
                comm,
                &requests[i]);
    }
    if (recvOffsets[i + 1] > recvOffsets[i]) {
      MPI_Irecv(targetReordered + recvOffsets[i] * typesize,
                recvOffsets[i + 1] - recvOffsets[i],
                datatype,
                i,
                tag,
                comm,
                &requests[static_cast<std::size_t>(commsize) + i]);
    }
  }
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::size_t i = 0; i < recvReorder.size(); ++i) {
    std::memcpy(targetChar + i * typesize, targetReordered + recvReorder[i] * typesize, typesize);
  }

  std::free(sourceReordered);
  std::free(targetReordered);
}
} // namespace seissol::io::reader
