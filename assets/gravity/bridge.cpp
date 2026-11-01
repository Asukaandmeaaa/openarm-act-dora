// Thin C ABI around the byte-for-byte frozen collection KDL implementation.
#ifdef COLLECTION_REFERENCE
#include "openarm_hardware/dynamics.hpp"
#else
#include <kdl/chaindynparam.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf_parser/urdf_parser.h>
#include <fstream>
#include <sstream>
#endif
#include <cmath>
#include <exception>
#include <memory>
#include <cstdio>

#ifndef COLLECTION_REFERENCE
// Same chain and KDL JntToGravity algorithm as collection. Direct URDF XML
// parsing avoids loading unrelated SDF plugins from the host ROS installation.
class StandaloneDynamics {
 public:
  StandaloneDynamics(std::string file, std::string base, std::string tip)
    : file_(file), base_(base), tip_(tip) {}
  bool Init() {
    std::ifstream file(file_); if (!file) return false;
    std::stringstream buffer; buffer << file.rdbuf();
    auto model=urdf::parseURDF(buffer.str());
    KDL::Tree tree;
    if (!model || !kdl_parser::treeFromUrdfModel(*model,tree) ||
        !tree.getChain(base_,tip_,chain_)) return false;
    solver_=std::make_unique<KDL::ChainDynParam>(chain_,KDL::Vector(0,0,-9.81));
    return true;
  }
  size_t JointCount() const { return chain_.getNrOfJoints(); }
  void GetGravity(const double* positions, double* torques) {
    KDL::JntArray q(JointCount()), tau(JointCount());
    for (size_t i=0;i<JointCount();++i) q(i)=positions[i];
    if (solver_->JntToGravity(q,tau)<0) throw std::runtime_error("JntToGravity failed");
    for (size_t i=0;i<JointCount();++i) torques[i]=tau(i);
  }
 private:
  std::string file_,base_,tip_;
  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainDynParam> solver_;
};
using Dynamics=StandaloneDynamics;
#else
using Dynamics=openarm_hardware::Dynamics;
#endif

extern "C" {
void* gravity_create(const char* urdf, const char* base, const char* tip) noexcept {
  try {
    auto p = std::make_unique<Dynamics>(urdf, base, tip);
    if (!p->Init() || p->JointCount() != 7) return nullptr;
    return p.release();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gravity_create: %s\n", e.what()); return nullptr;
  } catch (...) { return nullptr; }
}
int gravity_compute(void* ptr, const double* q, double* tau) noexcept {
  if (!ptr || !q || !tau) return -1;
  for (int i=0; i<7; ++i) if (!std::isfinite(q[i])) return -2;
  try {
    static_cast<Dynamics*>(ptr)->GetGravity(q, tau);
    for (int i=0; i<7; ++i) if (!std::isfinite(tau[i])) return -3;
    return 0;
  } catch (...) { return -4; }
}
void gravity_destroy(void* ptr) noexcept {
  delete static_cast<Dynamics*>(ptr);
}
}
