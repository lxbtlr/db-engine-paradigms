#pragma once
#include "common/runtime/Import.hpp"
#include <gtest/gtest.h>

class SSB : public ::testing::Test {
public:
  static runtime::Database& getDB() {
    static runtime::Database ssb;
    if (!ssb.hasRelation("lineorder")) {
      importSSB(std::string(DATADIR) + "/ssb/sf1/", ssb);
    }
    return ssb;
  }
};
