#include <vector>

#include "caffe/layers/interleave_layer.hpp"

#include <cstdio>

namespace caffe
{
  template <typename Dtype>
  __global__ void InterleaveForward( const int    nthreads
                                   , const Dtype* in_a
                                   , const Dtype* in_b
                                   , const Dtype* in_c 
                                   , const Dtype* in_d
                                   , const int    width
                                   , Dtype*       out_data )
  {
    CUDA_KERNEL_LOOP( index, nthreads )
    {
      int is_row_even = ( index / width ) % 2;
      int is_col_even = ( index % width ) % 2;
      if( !is_row_even )
      {
        
        if( !is_col_even )
        {
          //printf("in: %i, a_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ;
          out_data[index] = in_a[ ( index % width ) / 2 + ( index / width ) * 2 ]; 
        }
	else
        {
          //printf("in: %i, b_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ; 
          out_data[index] = in_b[ ( index % width ) / 2 + ( index / width ) * 2 ];
        }
      }
      else
      {
        if( !is_col_even )
        {        
          //printf("in: %i, c_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ; 
          out_data[index] = in_c[ ( index % width ) / 2 + ( index / width ) * 2 - 2 ];
        }
        else
        {
          //printf("in: %i, d_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ;
          out_data[index] = in_d[ ( index % width ) / 2 + ( index / width ) * 2 - 2]; 
        }
      }//if( !is_row_even )      
    }//CUDA_KERNEL_LOOP
  } 
 
  template <typename Dtype>
  void InterleaveLayer<Dtype>::Forward_gpu( const vector<Blob<Dtype>*>& bottom
                                          , const vector<Blob<Dtype>*>& top )
  {
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int width = top[0]->shape(3);
    const int count = top[0]->count();

    //printf( "w: %i\n", width );

    const Dtype* bottom_data_a = bottom[0]->gpu_data();
    const Dtype* bottom_data_b = bottom[1]->gpu_data();
    const Dtype* bottom_data_c = bottom[2]->gpu_data();
    const Dtype* bottom_data_d = bottom[3]->gpu_data();

    InterleaveForward<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        ( count, bottom_data_a, bottom_data_b, bottom_data_c, bottom_data_d, width, top_data );

    CUDA_POST_KERNEL_CHECK;
    
  }

  template <typename Dtype>
  __global__ void InterleaveBackward( const int nthreads
                                    , const Dtype* in_diff
                                    , const int width
                                    , Dtype* out_diff_a
                                    , Dtype* out_diff_b
                                    , Dtype* out_diff_c
                                    , Dtype* out_diff_d )
  {
    CUDA_KERNEL_LOOP( index, nthreads )
    {
      int is_row_even = ( index / width ) % 2;
      int is_col_even = ( index % width ) % 2;
      if( !is_row_even )
      {
        
        if( !is_col_even )
        {
          //printf("in: %i, a_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ;
          out_diff_a[ ( index % width ) / 2 + ( index / width ) * 2 ] = in_diff[index]; 
        }
	else
        {
          //printf("in: %i, b_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ; 
          out_diff_b[ ( index % width ) / 2 + ( index / width ) * 2 ] = in_diff[index];
        }
      }
      else
      {
        if( !is_col_even )
        {        
          //printf("in: %i, c_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ; 
          out_diff_c[ ( index % width ) / 2 + ( index / width ) * 2 - 2 ] = in_diff[index];
        }
        else
        {
          //printf("in: %i, d_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ;
          out_diff_d[ ( index % width ) / 2 + ( index / width ) * 2 - 2] = in_diff[index]; 
        }
      }//if
    }
  }

  template <typename Dtype>
  void InterleaveLayer<Dtype>::Backward_gpu( const vector<Blob<Dtype>*>& top
                                           , const vector<bool>& propagate_down
                                           , const vector<Blob<Dtype>*>& bottom )
  {
    //if( propagate_down[0] )
    //{
      const Dtype* top_diff = top[0]->gpu_diff();
      const int count	    = top[0]->count();
      const int width       = top[0]->shape(3);
  
      Dtype* bottom_diff_a = bottom[0]->mutable_gpu_diff();
      Dtype* bottom_diff_b = bottom[1]->mutable_gpu_diff();
      Dtype* bottom_diff_c = bottom[2]->mutable_gpu_diff();
      Dtype* bottom_diff_d = bottom[3]->mutable_gpu_diff(); 

      InterleaveBackward<Dtype>
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
        ( count, top_diff, width, bottom_diff_a, bottom_diff_b, bottom_diff_c, bottom_diff_d );

      CUDA_POST_KERNEL_CHECK; 
    //}
  }



















  INSTANTIATE_LAYER_GPU_FUNCS(InterleaveLayer);
  
} //namespace caffe
