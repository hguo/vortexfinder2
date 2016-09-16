#include "common/Inclusions.h"
#include <rocksdb/db.h>
#include <iostream>

int main(int argc, char **argv)
{
  if (argc < 3) return 1;
  
  rocksdb::DB* db;
  rocksdb::Options options;
  rocksdb::Status status = rocksdb::DB::Open(options, argv[1], &db);

  Inclusions inc;
  inc.ParseFromTextFile(argv[2]);
 
  std::string buf;
  diy::serialize(inc, buf);
  db->Put(rocksdb::WriteOptions(), "inclusions", buf);
  delete db;
  
  return 0;
}
