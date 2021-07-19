#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <cstring>
#include <numpy/arrayobject.h>
#include <vector>
#include <random>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <limits>
#include <memory>
#include "sepia/source/sepia.hpp"

#ifndef FILTER_HPP_
#define FILTER_HPP_

using Event = sepia::event<sepia::type::dvs>;

class Mask{
private:
    uint16_t x_;
    uint16_t y_;
    uint16_t w_;
public:
    std::vector<uint16_t> min_x_, max_x_, min_y_, max_y_;
    Mask(){
        x_=0;y_=0;w_=0;
    }
    Mask(uint16_t& x, uint16_t& y, uint16_t& w){
        x_ = x;
        y_ = y;
        uint32_t nb_pix = x_ * y_;
        uint32_t i, j;
        min_x_.resize(nb_pix);min_y_.resize(nb_pix);
        max_x_.resize(nb_pix);max_y_.resize(nb_pix);
        for(uint32_t k = 0; k < nb_pix; k++){
            i = k % x_;j = k / x_;
            min_y_.at(k) = std::max<int32_t>(0, j - w);
            max_y_.at(k) = std::min<int32_t>(y_, j + w);
            min_x_.at(k) = std::max<int32_t>(0, i - w);
            max_x_.at(k) = std::min<int32_t>(x_, i + w);
        }
    };
};

class Filter{
    private:
        std::vector<uint64_t> timesurface_;
        uint64_t th_time_;
        uint16_t th_ev_;
        uint16_t w_;
        uint16_t x_;
        uint16_t y_;
        Mask M_;
    public:
        Filter(uint16_t& x, uint16_t& y, uint16_t& w, uint64_t& time_th, uint16_t& ev_th){
            timesurface_.resize(x * y, std::numeric_limits<uint64_t>::max());
            x_ = x;
            y_ = y;
            w_ = w;
            th_time_ = time_th;
            th_ev_ = ev_th;
            M_ = Mask(x_, y_, w_);
        };
        uint32_t getPos(const uint16_t& x, const uint16_t& y){return y * x_ + x;}
        void process(std::vector<Event>& in, std::vector<Event>& out){
            out.reserve(in.size());
            uint16_t x, y, ct;
            uint32_t p, p_n;
            uint64_t t_min;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                timesurface_.at(p) = it->t;
                ct = 0;
                t_min = (it->t > th_time_) ? it->t - th_time_ : 0;
                for(x = M_.min_x_.at(p); x < M_.max_x_.at(p); x++){
                    for(y = M_.min_y_.at(p); y < M_.max_y_.at(p); y++){
                        p_n = getPos(x, y);             
                        ct += (timesurface_.at(p_n) >= t_min && timesurface_.at(p_n) <= it->t) ? 1 : 0;
                    }
                }
                if(ct > th_ev_)
                    out.push_back(Event(*it));
            }
        };
};


#endif 