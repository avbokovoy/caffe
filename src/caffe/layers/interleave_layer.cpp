// Interleave operation

#include <vector>

#include "caffe/layers/interleave_layer.hpp"

#include <iostream>

namespace caffe
{
  template<typename Dtype>
  void InterleaveLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*>& bottom
                                         , const vector<Blob<Dtype>*>& top ){}
  
  
  template <typename Dtype>
  void InterleaveLayer<Dtype>::Reshape( const vector<Blob<Dtype>*>& bottom
                                      , const vector<Blob<Dtype>*>& top )
  {
    for( int i = 1; i < bottom.size(); ++i )
    {
      CHECK(  bottom[0]->shape() == bottom[i]->shape() 
           && bottom[0]->shape(3) == bottom[i]->shape(3)
           && bottom[0]->shape(2) == bottom[i]->shape(2) )
        << "bottom[0]: " << bottom[0]->shape_string()
        << ", bottom[" << i << "]: " << bottom[i]->shape_string();
    }

    vector<int> new_shape;
    new_shape.push_back( bottom[0]->shape(0) );
    new_shape.push_back( bottom[0]->shape(1) );
    new_shape.push_back( 2 * bottom[0]->shape(2) );
    new_shape.push_back( 2 * bottom[0]->shape(3) );

    top[0]->Reshape( new_shape );
  }

  template <typename Dtype>
  void InterleaveLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom
                                          , const vector<Blob<Dtype>*>& top )
  {
    const int count  = top[0]->count();
    const int width  = top[0]->shape(3);
    Dtype* top_data  = top[0]->mutable_cpu_data();

    const Dtype* bottom_data_a = bottom[0]->cpu_data();
    const Dtype* bottom_data_b = bottom[1]->cpu_data();
    const Dtype* bottom_data_c = bottom[2]->cpu_data();
    const Dtype* bottom_data_d = bottom[3]->cpu_data();

    int offset_width = 0;  
    for( int index = 0; index < count ; ++index )
    {
      int is_row_even = ( index / width ) % 2;
      int is_col_even = ( index % width ) % 2;
      if( !is_row_even )
      {
        
        if( !is_col_even )
        {
          //printf("in: %i, a_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ;
          top_data[index] = bottom_data_a[ ( index % width ) / 2 + ( index / width ) * 2 ]; 
        }
	else
        {
          //printf("in: %i, b_%i\n", index, ( index % width ) / 2 + index / width * 2 ) ; 
          top_data[index] = bottom_data_b[ ( index % width ) / 2 + ( index / width ) * 2 ];
        }
      }
      else
      {
        if( !is_col_even )
        {        
          //printf("in: %i, c_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ; 
          top_data[index] = bottom_data_c[ ( index % width ) / 2 + ( index / width ) * 2 - 2 ];
        }
        else
        {
          //printf("in: %i, d_%i\n", index, ( index % width ) / 2 + index / width * 2 - 2) ;
          top_data[index] = bottom_data_d[ ( index % width ) / 2 + ( index / width ) * 2 - 2]; 
        }
      }
      /*if( !( i % width ) && i  > 0 )
      {
         i += width;
         offset_width += width;
      }

      top_data[i]     = bottom_data_a[ ( i - offset_width )/ 2 ];
      top_data[i + 1] = bottom_data_b[ ( i - offset_width )/ 2 ];
      top_data[i + width]     = bottom_data_c[ ( i - offset_width )/ 2 ];
      top_data[i + width + 1] = bottom_data_d[ ( i - offset_width )/ 2 ];*/
    }    
  }

  template <typename Dtype>
  void InterleaveLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*>& top
                                           , const vector<bool>& propagate_down
                                           , const vector<Blob<Dtype>*>& bottom )
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    const int count	  = top[0]->count();
    const int width       = top[0]->shape(3);
  
    Dtype* bottom_diff_a = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff_b = bottom[1]->mutable_cpu_diff();
    Dtype* bottom_diff_c = bottom[2]->mutable_cpu_diff();
    Dtype* bottom_diff_d = bottom[3]->mutable_cpu_diff();

    int offset_width = 0;
    for( int i = 0; i < count - width; i += 2 )
    {
      if( !( i % width ) && i  > 0 )
      {
         i += width;
         offset_width += width;
      }

      if( propagate_down[0] )
      {
        bottom_diff_a[ ( i - offset_width )/ 2 ] = top_diff[i];
      //}
      //if( propagate_down[i + 1] )
      //{
        bottom_diff_b[ ( i - offset_width )/ 2 ] = top_diff[i + 1];
      //}
      //if( propagate_down[i + width] )
      //{
        bottom_diff_c[ ( i - offset_width )/ 2 ] = top_diff[i + width];
      //}
      //if( propagate_down[i + width + 1] )
      //{
        bottom_diff_d[ ( i - offset_width )/ 2 ] = top_diff[i + width + 1 ];
      }
    }     
  }

#ifdef CPU_ONLY
  STUB_GPU(InterleaveLayer);
#endif

  INSTANTIATE_CLASS(InterleaveLayer);
  REGISTER_LAYER_CLASS(Interleave);
} //namespace caffe
