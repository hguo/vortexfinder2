#include "def.h"
#include "common/VortexTransition.h"
#include <cstdio>

#ifdef WITH_PROTOBUF
#include "common/DataInfo.pb.h"
#endif

#if WITH_ROCKSDB
#include <rocksdb/db.h>

int main(int argc, char **argv)
{
  if (argc < 2) return 1;

  rocksdb::DB* db;
  rocksdb::Options options;
  rocksdb::Status status = rocksdb::DB::Open(options, argv[1], &db);

  VortexTransition vt;
  vt.LoadFromDB(db);
  // vt.ConstructSequence();
  vt.PrintSequence();

  delete db;
  return 0;
}
#else
int main(int argc, char **argv)
{
  if (argc < 4) {
    fprintf(stderr, "Usage: %s <dataname> <ts> <tl>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const std::string dataname = argv[1];
  const int ts = atoi(argv[2]), 
            tl = atoi(argv[3]);

  VortexTransition vt;
  vt.LoadFromFile(dataname, ts, tl);
  vt.ConstructSequence();
  vt.PrintSequence();

  return 0;
}
#endif
