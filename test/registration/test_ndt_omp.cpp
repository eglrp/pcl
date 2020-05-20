/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2018-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include <pcl/test/gtest.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/ndt_omp.h>

using namespace pcl;
using namespace pcl::io;

PointCloud<PointXYZ> cloud_source, cloud_target;

template<typename NDT>
Eigen::Matrix4f test(NDT& reg)
{
  using PointT = PointNormal;
  PointCloud<PointT>::Ptr src(new PointCloud<PointT>);
  copyPointCloud(cloud_source, *src);
  PointCloud<PointT>::Ptr tgt(new PointCloud<PointT>);
  copyPointCloud(cloud_target, *tgt);
  PointCloud<PointT> output;

  reg.setStepSize(0.05);
  reg.setResolution(0.025f);
  reg.setInputSource(src);
  reg.setInputTarget(tgt);
  reg.setMaximumIterations(50);
  reg.setTransformationEpsilon(1e-8);
  // Register
  reg.align(output);
  EXPECT_EQ(int(output.points.size()), int(cloud_source.points.size()));
  EXPECT_LT(reg.getFitnessScore(), 0.001);

  return reg.getFinalTransformation();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PCL, NormalDistributionsTransformOMP_KDTREE)
{
  NormalDistributionsTransformOMP<PointNormal, PointNormal> reg_omp;
  reg_omp.setNeighborhoodSearchMethod(pcl::KDTREE);
  Eigen::Matrix4f trans_omp = test(reg_omp);

  NormalDistributionsTransform<PointNormal, PointNormal> reg;
  Eigen::Matrix4f trans = test(reg);

  // The result of NDT_OMP(KDTREE) must be identical to the original NDT's result
  double err = (trans - trans_omp).array().abs().maxCoeff();
  EXPECT_LT(err, 1e-6);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PCL, NormalDistributionsTransformOMP_DIRECT7)
{
  NormalDistributionsTransformOMP<PointNormal, PointNormal> reg;
  reg.setNeighborhoodSearchMethod(pcl::DIRECT7);
  test(reg);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST(PCL, NormalDistributionsTransformOMP_DIRECT1)
{
  NormalDistributionsTransformOMP<PointNormal, PointNormal> reg;
  reg.setNeighborhoodSearchMethod(pcl::DIRECT1);
  test(reg);
}

int
main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "No test files given. Please download `bun0.pcd` and `bun4.pcd`pass "
                 "their path to the test."
              << std::endl;
    return (-1);
  }

  if (loadPCDFile(argv[1], cloud_source) < 0) {
    std::cerr << "Failed to read test file. Please download `bun0.pcd` and pass its "
                 "path to the test."
              << std::endl;
    return (-1);
  }
  if (loadPCDFile(argv[2], cloud_target) < 0) {
    std::cerr << "Failed to read test file. Please download `bun4.pcd` and pass its "
                 "path to the test."
              << std::endl;
    return (-1);
  }

  testing::InitGoogleTest(&argc, argv);
  return (RUN_ALL_TESTS());
}
