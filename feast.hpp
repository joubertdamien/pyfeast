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
float_t closeId(float_t& th, float_t& th_close){
    return th + th_close;
}
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
        float_t getVal(uint64_t& dt){
            return (dt >= max_dt_) ? 0 : lut_.at(dt / res_t_);
        }
};
template<auto CloseF>
class FeastNeuron{
    private:
        uint16_t w_;       // half size of the local context
        uint16_t size_;    //  size in pixels
        uint64_t tau_;     // time constant of the time surface
        float_t th_open_;  // threshold opening if nobody reacts
        float_t th_close_; // threshold closure if neuron selected
        float_t th_;       //  Current Threshold
        float_t lr_;       // learning rate of the feature
        float_t ilr_;      // 1 - learning rate of the feature
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
        void init(){
            th_ = 1;
            std::default_random_engine generator;
            std::uniform_real_distribution<float_t> distribution(0.0,1.0);
            weights_.resize(size_);
            for(int i = 0; i < size_; i++)
                weights_.at(i) = distribution(generator);
            n2_ = 0;
            std::for_each(weights_.begin(), weights_.end(), [&](float_t &v){n2_ += v*v;});
            n2_ = std::sqrt(n2_);
            std::for_each(weights_.begin(), weights_.end(), [&](float_t &v){v /= n2_;});
        };
        uint16_t getsize(){return (2 * w_ + 1) * (2 * w_ + 1);}
        void match(std::vector<float>& ctx, float_t& res, bool& activated){
            res = 0;
            for (uint16_t i = 0; i < size_; i++){
                res += ctx.at(i) * weights_.at(i); 
            }
            activated = (res > th_) ? 1 : 0;
        }
        void open(){th_ = std::max<float>(th_ - th_open_, 0);}
        void close(){th_ = std::min<float>(CloseF(th_, th_close_), 1);}
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
        float_t getEntropy(){
            std::vector<float_t> histo(11, 0.0);
            for (uint16_t i = 0; i < size_; i++){
                histo.at(static_cast<uint8_t>(10*weights_.at(i))) ++;
            }
            std::for_each(histo.begin(), histo.end(), [&](float_t &v){v/=11;});
            float_t entro = 0.0;
            for(uint16_t i = 0; i < 10; i++)
                entro += (histo.at(i) > 0) ? log(histo.at(i)) * histo.at(i) : 0;
            return -entro;
        }
};
template<class Neuron>
class FeastLayer{
    private:
        std::vector<uint64_t> timesurface_;
        std::vector<uint16_t> indexsurface_;
        std::vector<Neuron> neurons_;
        std::vector<float_t> ev_ctx_;
        std::vector<bool> match_;
        std::vector<float_t> res_;
        uint16_t x_;      // number of rows
        uint16_t y_;      // number of columns
        uint16_t w_;      // half size of the local context
        uint16_t hw_;     // half size of the ROI
        uint16_t sroi_;   // size of the ROI
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
            indexsurface_.resize(x * y, 0);
            x_ = x;
            y_ = y;
            w_ = w;
            hw_ = 2 * w + 1;
            sroi_= hw_ * hw_;
            M_ = Mask(x_, y_, w_);
            tau_ = tau;
            th_open_=th_open;
            th_close_ = th_close;
            lr_ = lr;
            nb_ = nb;
            initNeuron();
            lut_ = ExpLUT(tau_, 1024);
            ev_ctx_.resize(sroi_);
            match_.resize(sroi_);
            res_.resize(sroi_);
        };
        void initNeuron(){
            neurons_.reserve(nb_);
            for(int i = 0; i < nb_; i++){
                neurons_.push_back(Neuron(w_, tau_, th_open_, th_close_, lr_));
            }
        }
        uint32_t getPos(const uint16_t& x, const uint16_t& y){return y * x_ + x;}
        uint32_t getPosCtx(const uint16_t& x, const uint16_t& y){return y * hw_ + x;}
        void getWeights(std::vector<float_t>& weigths){
            uint16_t s = nb_ * sroi_;
            weigths.resize(s);
            for (int i = 0; i < s; i++){
                weigths.at(i) = neurons_.at(i / sroi_).getWeight(i % sroi_);
            }
        }
        void getThs(std::vector<float_t>& ths){
            ths.resize(nb_);
            for (int i = 0; i < nb_; i++){
                ths.at(i) = neurons_.at(i).getTh();
            }
        }
        void getEnt(std::vector<float_t>& ent){
            ent.resize(nb_);
            for (int i = 0; i < nb_; i++){
                ent.at(i) = neurons_.at(i).getEntropy();
            }
        }
        void process(std::vector<Event>& in, std::vector<NeuronEvent>& out){
            uint32_t p;
            uint16_t id_winner;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                timesurface_.at(p) = it->t;
                getTimeCtx(ev_ctx_, p, it->t);
                matchEventneuron(ev_ctx_, match_, res_, id_winner);
                applyFEAST(id_winner, out, ev_ctx_, it);
            }
        };
        void processIndex(std::vector<Event>& in, std::vector<NeuronEvent>& out){
            uint32_t p;
            uint16_t id_winner;
            for(auto it = in.begin(); it < in.end(); it++){
                p = getPos(it->x, it->y);
                getIndexCtx(ev_ctx_, p, it->t);
                matchEventneuron(ev_ctx_, match_, res_, id_winner);
                applyFEAST(id_winner, out, ev_ctx_, it);
            }
        };
        void getTimeCtx(std::vector<float_t>& localCtx, uint32_t pos, uint64_t t){
            uint16_t x, y;
            uint32_t pctx = 0, p;
            uint64_t dt;
            float sum = 0;
            for(x = M_.min_x_.at(pos); x <= M_.max_x_.at(pos); x++){
                for(y = M_.min_y_.at(pos); y <= M_.max_y_.at(pos); y++){
                    p = getPos(x, y);  
                    dt = t - timesurface_.at(p);     
                    localCtx.at(pctx) = lut_.getVal(dt);
                    sum += localCtx.at(pctx);
                    pctx += 1;
                }
            }
            if(sum > 0)
                std::for_each(localCtx.begin(), localCtx.end(), [&](float_t &v){v/=sum;});
        };
        void getIndexCtx(std::vector<float_t>& localCtx, uint32_t pos, uint64_t t){
            uint16_t x, y;
            uint32_t pctx = 0, p;
            for(x = M_.min_x_.at(pos); x <= M_.max_x_.at(pos); x++){
                for(y = M_.min_y_.at(pos); y <= M_.max_y_.at(pos); y++){
                    p = getPos(x, y); 
                    if(p == pos)
                       indexsurface_.at(p) = sroi_;
                    else{
                        if(indexsurface_.at(p) > 0)
                            indexsurface_.at(p)--;
                    } 
                    localCtx.at(pctx) = static_cast<float_t>(indexsurface_.at(p))/static_cast<float_t>(sroi_);
                    pctx += 1;
                }
            }
        };
        void matchEventneuron(std::vector<float_t>& localCtx, std::vector<bool>& match, std::vector<float_t>& res, uint16_t& id){
            bool m; float_t r, m_r = 0;
            id = std::numeric_limits<uint16_t>::max();
            for(uint16_t i = 0; i < nb_; i++){
                neurons_.at(i).match(localCtx, r, m);
                match.at(i) = m;
                res.at(i) = r; 
                if(res.at(i) > m_r && match_.at(i) == 1){
                    m_r = res.at(i);
                    id = i;
                }
            }
        }
        void applyFEAST(uint16_t& id, std::vector<NeuronEvent>& out, std::vector<float_t>& localCtx, std::vector<Event>::iterator& it){
            if( id > nb_)
                std::for_each(neurons_.begin(), neurons_.end(), [&](Neuron& n){n.open();});
            else{
                neurons_.at(id).close();
                neurons_.at(id).learn(localCtx);
                out.push_back(NeuronEvent{it->t, it->x, it->y, id});
            }
        }
};
#endif 