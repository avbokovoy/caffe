#ifndef CAFFE_INTERLEAVE_LAYER_HPP_
#define CAFFE_INTERLEAVE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
  /**
   *@brief Compute interleave of 4 input blobs as in https://arxiv.org/pdf/1606.00373.pdf
   */

  template <typename Dtype>
  class InterleaveLayer : public Layer<Dtype>
  {
    public: 
      explicit InterleaveLayer( const LayerParameter& param ) : Layer<Dtype>(param){}
      virtual void LayerSetUp( const vector<Blob<Dtype>*>& bottom
                             , const vector<Blob<Dtype>*>& top );
      virtual void Reshape( const vector<Blob<Dtype>*>& bottom
                          , const vector<Blob<Dtype>*>& top );

      virtual inline const char* type() const {return "Interleave";}
      virtual inline int ExactNumBottomBlobs() const { return 4; }
      virtual inline int ExactNumTopBlobs() const { return 1; }
    
    protected:
      virtual void Forward_cpu( const vector<Blob<Dtype>*>& bottom
                              , const vector<Blob<Dtype>*>& top );
      virtual void Forward_gpu( const vector<Blob<Dtype>*>& bottom
                              , const vector<Blob<Dtype>*>& top );
      virtual void Backward_cpu( const vector<Blob<Dtype>*>& top
                               , const vector<bool>& propagate_down
                               , const vector<Blob<Dtype>*>& bottom );
      virtual void Backward_gpu( const vector<Blob<Dtype>*>& top
                               , const vector<bool>& propagate_down
                               , const vector<Blob<Dtype>*>& bottom );



  }; 



} // namespace caffe


#endif //CAFFE_INTERLEAVE_LAYER_HPP_
