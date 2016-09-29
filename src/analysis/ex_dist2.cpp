#include "def.h"
#include "common/VortexTransition.h"
#include <cstdio>

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
