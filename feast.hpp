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
struct __attribute__((__packed__)) NeuronEvent{
    uint64_t t;
    uint16_t x;
    uint16_t y;
    uint16_t neuron_id;
};
class Mask{
    private:
        uint16_t x_;
        uint16_t y_;
        uint16_t w_;
    public:
        std::vector<uint16_t> min_x_, max_x_, min_y_, max_y_;
        Mask(){x_=0;y_=0;w_=0;};
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
class ExpLUT{
    private:
        uint64_t tau_;
        uint64_t max_dt_;
        uint16_t res_;
        uint64_t res_t_;
        std::vector<float> lut_;
    public:
        ExpLUT(){};
        ExpLUT(uint64_t tau, uint16_t res){
            res_ = res;
            tau_= tau;
            max_dt_ = 3 * tau_;
            lut_.resize(res_);
            res_t_ = max_dt_ / res;
            max_dt_ = res * res_t_; 
            for(uint16_t i = 0; i < res_; i++){
                lut_.at(i) = std::exp(- static_cast<float_t>(i * res_t_) / tau);
            }
        }
        float_t getVal(uint64_t dt){
            return (dt >= max_dt_) ? 0 : lut_.at(dt / res_t_);
        }
};
class FeastNeuron{
    private:
        uint16_t w_;       // half size of the local context
        uint16_t size_;    //  size in pixels
        uint64_t tau_;     // time constant of the time surface
        float_t th_open_;  // threshold opening if nobody reacts
        float_t th_close_; // threshold closure if neuron selected
        float_t th_;       //  Current Threshold
        float_t lr_;       // learning rate of the feature
        float_t ilr_;      // 1-learning rate of the feature
        float_t n2_;       // n2 norm
        std::vector<float_t> weights_; 
    public:
        FeastNeuron(){};
        FeastNeuron(uint16_t& w, uint64_t& tau, float_t& th_open, float_t& th_close, float_t& lr){
            w_ = w;
            tau_ = tau;
            th_open_=th_open;
            th_close_ = th_close;
            lr_ = lr;
            ilr_ = 1 - lr;
            size_ = getsize();
            init();
        };
        ~FeastNeuron(){};
        void init(){
            th_ = 1;
            std::default_random_engine generator;
            std::uniform_real_distribution<float_t> distribution(0.0,1.0);
            weights_.resize(size_);
            for(int i = 0; i < size_; i++)
                weights_.at(i) = distribution(generator);
            n2_ = 0;
            std::for_each(weights_.begin(), weights_.end(), [&](float_t &v){n2_+=v*v;});
            n2_ = std::sqrt(n2_);
            std::for_each(weights_.begin(), weights_.end(), [&](float_t &v){v/=n2_;});
        };
        uint16_t getsize(){return (2 * w_ + 1) * (2 * w_ + 1);}
        void match(std::vector<float>& ctx, float_t& res, bool& activated){
            res = 0;
            for (uint16_t i = 0; i < size_; i++){
                res += ctx.at(i) * weights_.at(i); 
            }
            activated = (res < th_) ? 1 : 0;
        }
        void open(){th_ = std::min<float>(th_ + th_open_, 1);}
        void close(){th_ = std::max<float>(th_ - th_close_, 0);}
        void learn(std::vector<float>& ctx){
            n2_ = 0;
            for (uint16_t i = 0; i < size_; i++){
                weights_.at(i) = lr_ * ctx.at(i) + ilr_* weights_.at(i); 
                n2_ += weights_.at(i) * weights_.at(i);
            }
            n2_ = std::sqrt(n2_);
            std::for_each(weights_.begin(), weights_.end(), [&](float_t &v){v/=n2_;});
        }
        float_t getWeight(uint16_t p){return weights_.at(p);}
        float_t getTh(){return th_;}
};
class FeastLayer{
    private:
        std::vector<uint64_t> timesurface_;
        std::vector<FeastNeuron> neurons_;
        std::vector<float_t> ev_ctx_;
        std::vector<bool> match_;
        std::vector<float_t> res_;
        uint16_t x_;      // number of rows
        uint16_t y_;      // number of columns
        uint16_t w_;      // half size of the local context
        uint16_t hw_;     // half size of the ROI
        uint64_t tau_;    // time constant of the time surface
        float_t th_open_; // threshold opening if nobody reacts
        float_t th_close_;// threshold closure if neuron selected
        float_t lr_;      // learning rate of tshe feature
        uint16_t nb_;     // number of neurons
        Mask M_;          // mask used for the layer to map the previous layer 
        ExpLUT lut_;
    public:
        FeastLayer(uint16_t& x, uint16_t& y, uint16_t& w, uint64_t& tau, float_t& th_open, float_t& th_close, float_t& lr, uint16_t& nb){
            timesurface_.resize(x * y, std::numeric_limits<uint64_t>::max());
            x_ = x;
            y_ = y;
            w_ = w;
            hw_ = 2 * w + 1;
            M_ = Mask(x_, y_, w_);
            tau_ = tau;
            th_open_=th_open;
            th_close_ = th_close;
            lr_ = lr;
            nb_ = nb;
            initNeuron();
            lut_ = ExpLUT(tau_, 1024);
            ev_ctx_.resize(getSizeCtx());
            match_.resize(getSizeCtx());
            res_.resize(getSizeCtx());
        };
        void initNeuron(){
            neurons_.reserve(nb_);
            for(int i = 0; i < nb_; i++){
                neurons_.push_back(FeastNeuron(w_, tau_, th_open_, th_close_, lr_));
            }
        }
        uint32_t getPos(const uint16_t& x, const uint16_t& y){return y * x_ + x;}
        uint32_t getPosCtx(const uint16_t& x, const uint16_t& y){return y * hw_ + x;}
        uint16_t getSizeCtx(){return hw_ * hw_;}
        void getWeights(std::vector<float_t>& weigths){
            uint16_t s = getSizeCtx();
            weigths.resize(nb_ * s);
            for (int i = 0; i < nb_ * s; i++){
                weigths.at(i) = neurons_.at(i / s).getWeight(i % s);
            }
        }
        void getThs(std::vector<float_t>& ths){
            ths.resize(nb_);
            for (int i = 0; i < nb_; i++){
                ths.at(i) = neurons_.at(i).getTh();
            }
        }
        void process(std::vector<Event>& in, std::vector<NeuronEvent>& out){
            uint32_t p;
            uint16_t id_winner;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                timesurface_.at(p) = it->t;
                getEventCtx(ev_ctx_, p, it->t);
                matchEventneuron(ev_ctx_, match_, res_);
                getResFEAST(match_, res_, id_winner);
                applyFEAST(id_winner, out, ev_ctx_, it);
            }
        };
        void getEventCtx(std::vector<float_t>& localCtx, uint32_t pos, uint64_t t){
            uint16_t x, y;
            uint32_t pctx, p;
            float sum = 0;
            for(x = M_.min_x_.at(pos); x <= M_.max_x_.at(pos); x++){
                for(y = M_.min_y_.at(pos); y <= M_.max_y_.at(pos); y++){
                    pctx = getPosCtx(x - M_.min_x_.at(pos), y - M_.min_y_.at(pos));
                    p = getPos(x, y);             
                    localCtx.at(pctx) = lut_.getVal(t - timesurface_.at(p));
                    sum += localCtx.at(pctx);
                }
            }
            if(sum > 0)
                std::for_each(localCtx.begin(), localCtx.end(), [&](float_t &v){v/=sum;});
        };
        void matchEventneuron(std::vector<float_t>& localCtx, std::vector<bool>& match, std::vector<float_t>& res){
            bool m; float_t r;
            for(uint16_t i = 0; i < nb_; i++){
                neurons_.at(i).match(localCtx, r, m);
                match.at(i) = m;
                res.at(i) = r; 
            }
        }
        void getResFEAST(std::vector<bool>& match, std::vector<float_t>& res, uint16_t& id){
            float_t m_v = 1;
            id = std::numeric_limits<uint16_t>::max();
            for(int i = 0; i < nb_; i++){
                if(res.at(i) < m_v && match_.at(i) == 1){
                    m_v = res.at(i);
                    id = i;
                }
            }
        }
        void applyFEAST(uint16_t& id, std::vector<NeuronEvent>& out, std::vector<float_t>& localCtx, std::vector<Event>::iterator& it){
            if( id > nb_)
                std::for_each(neurons_.begin(), neurons_.end(), [&](FeastNeuron& n){n.open();});
            else{
                neurons_.at(id).close();
                neurons_.at(id).learn(localCtx);
                out.push_back(NeuronEvent{it->t, it->x, it->y, id});
            }
        }
};
#endif 