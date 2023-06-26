/*
 * Software License Agreement
 *
 *  Point to plane metric for point cloud distortion measurement
 *  Copyright (c) 2017, MERL
 *
 *  All rights reserved.
 *
 *  Contributors:
 *    Dong Tian <tian@merl.com>
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
 */

#ifndef PCC_PROCESSING_HPP
#define PCC_PROCESSING_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <mutex>

using namespace std;

namespace pcc_processing {

  class PccPointCloud;

#define MAX_NUM_FIELDS 24
  class PointBaseSet
  {
  public:
    PointBaseSet() {};
    virtual int loadPoints( PccPointCloud *pPcc, long int idx ) = 0;
  };

  class PointXYZSet : public PointBaseSet
  {
  private:
    int idxInLine[3];           //! index of the x,y,z in the line
  public:
    vector< vector<double> > p;
    PointXYZSet() {};
    ~PointXYZSet();
    virtual int loadPoints( PccPointCloud *pPcc, long int idx );
    void init( long int size, int i0 = -1, int i1 = -1, int i2 = -1 );
  };

  class RGBSet : public PointBaseSet
  {
  private:
    int idxInLine[3];           //! index of the r,g,b in the line
  public:
    vector<vector<unsigned char>> c;
    RGBSet() {}
    ~RGBSet();
    virtual int loadPoints( PccPointCloud *pPcc, long int idx );
    void init( long int size, int i0 = -1, int i1 = -1, int i2 = -1 );
  };

  class NormalSet : public PointBaseSet
  {
  private:
    int idxInLine[3];           //! index of the x,y,z in the line
  public:
    vector<vector<float>> n;
    NormalSet() { }
    ~NormalSet();
    virtual int loadPoints( PccPointCloud *pPcc, long int idx );
    void init( long int size, int i0 = -1, int i1 = -1, int i2 = -1 );
  };

  class LidarSet : public PointBaseSet
  {
  private:
    int idxInLine[1];           //! index of the reflectance in the line
  public:
    vector<unsigned short> reflectance;
    LidarSet() { }
    ~LidarSet();
    virtual int loadPoints( PccPointCloud *pPcc, long int idx );
    void init( long int size, int i0 = -1 );
  };

  /**!
   * \brief
   *  Wrapper class to hold different type of attributes
   *  Easier to include new types of attributes
   *  
   *  Dong Tian <tian@merl.com>
   */
  class PccPointCloud
  {
  public:
    enum PointFieldTypes { INT8 = 1,
                           UINT8 = 2,
                           INT16 = 3,
                           UINT16 = 4,
                           INT32 = 5,
                           UINT32 = 6,
                           FLOAT32 = 7,
                           FLOAT64 = 8 };

    long int size;                 //! The number of points
    int fileFormat;                //! 0: ascii. 1: binary_little_endian
    int fieldType[MAX_NUM_FIELDS]; //! The field type
    int fieldPos [MAX_NUM_FIELDS]; //! The field position in the line memory
    int fieldNum;                  //! The number of field available
    int fieldSize;                 //! The memory size of the used fields
    int dataPos;                   //! The file pointer to the beginning of data
    int lineNum;                   //! The number of lines of header section
    unsigned char lineMem[MAX_NUM_FIELDS*sizeof(double)];

    int checkFile( string fileName );
    int checkField( string fileName, string fieldName, string fieldType1, string fieldType2 = "None", string fieldType3 = "None", string fieldType4 = "None" );
    int loadLine( ifstream &in );
#if _WIN32
    int seekBinary(ifstream &in);
    int seekAscii(ifstream &in);
#endif

  public:
    PointXYZSet xyz;
    RGBSet rgb;
    NormalSet normal;
    LidarSet lidar;

    bool bXyz;
    bool bRgb;
    bool bNormal;
    bool bLidar;

    PccPointCloud();
    ~PccPointCloud();
    int load( string inFile, bool isNormal = false );
  };

};

#endif
