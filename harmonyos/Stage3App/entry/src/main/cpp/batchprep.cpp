#include <time.h>
#include <napi/native_api.h>
#include <cstdint>
#include <cstring>
#include <cstddef>
static uint32_t h,w,ch,classes,batch;
static double means[3],stds[3];
static bool buf(napi_env env,napi_value v,void **p,size_t *n){return napi_get_arraybuffer_info(env,v,p,n)==napi_ok;}
static uint32_t integer(napi_env e,napi_value v){uint32_t x=0;napi_get_value_uint32(e,v,&x);return x;}
static napi_value init(napi_env e,napi_callback_info info){
 napi_value a[7];size_t n=7;napi_get_cb_info(e,info,&n,a,nullptr,nullptr);
 h=integer(e,a[0]);w=integer(e,a[1]);ch=integer(e,a[2]);classes=integer(e,a[3]);batch=integer(e,a[4]);
 void *m,*s;size_t mn,sn;if(n!=7||ch>3||!buf(e,a[5],&m,&mn)||!buf(e,a[6],&s,&sn)||mn!=ch*8||sn!=ch*8){napi_throw_error(e,nullptr,"Invalid preprocessing config");return nullptr;}
 memcpy(means,m,mn);memcpy(stds,s,sn);napi_value result;napi_get_undefined(e,&result);return result;
}
static double random01(uint32_t &s){s=1664525u*s+1013904223u;return static_cast<double>(s)/4294967296.0;}
static napi_value fill(napi_env e,napi_callback_info info){
 napi_value a[8];size_t n=8;napi_get_cb_info(e,info,&n,a,nullptr,nullptr);
 void *ip,*lp,*op,*xp,*yp;size_t il,ll,ol,xl,yl;
 if(n!=8||!buf(e,a[0],&ip,&il)||!buf(e,a[1],&lp,&ll)||!buf(e,a[2],&op,&ol)||!buf(e,a[3],&xp,&xl)||!buf(e,a[4],&yp,&yl)){napi_throw_error(e,nullptr,"Invalid batch buffers");return nullptr;}
 uint32_t start=integer(e,a[5]),seed=integer(e,a[6]);bool aug=false;napi_get_value_bool(e,a[7],&aug);
 const size_t area=h*w;const auto* images=static_cast<uint8_t*>(ip);const auto* labels=static_cast<uint8_t*>(lp);const auto* order=static_cast<int32_t*>(op);auto* x=static_cast<float*>(xp);auto* y=static_cast<float*>(yp);
 if(xl!=batch*ch*area*4||yl!=batch*classes*4||start+batch>ll||il!=ll*area*ch||(ol&&ol!=ll*4)){napi_throw_error(e,nullptr,"Batch shape mismatch");return nullptr;}
 memset(y,0,yl);
 for(uint32_t b=0;b<batch;b++){
  uint32_t idx=ol?order[start+b]:start+b;
  if(idx>=ll||labels[idx]>=classes){napi_throw_error(e,nullptr,"Index out of range");return nullptr;}
  y[b*classes+labels[idx]]=1;
  bool flip=aug&&random01(seed)<0.5;int dy=aug?static_cast<int>(random01(seed)*9)-4:0;int dx=aug?static_cast<int>(random01(seed)*9)-4:0;
  for(uint32_t c=0;c<ch;c++)for(uint32_t row=0;row<h;row++)for(uint32_t col=0;col<w;col++){
   int sy=static_cast<int>(row)+dy;int sx=static_cast<int>(flip?w-1-col:col)+dx;
   double v=(sy>=0&&sy<static_cast<int>(h)&&sx>=0&&sx<static_cast<int>(w))?static_cast<double>(images[(idx*area+sy*w+sx)*ch+c])/255.0:0.0;
   x[(b*ch+c)*area+row*w+col]=static_cast<float>((v-means[c])/stds[c]);
  }
 }
 napi_value result;napi_create_uint32(e,seed,&result);return result;
}
static napi_value monotonic(napi_env e,napi_callback_info info){ timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);napi_value result;napi_create_double(e,t.tv_sec*1000.0+t.tv_nsec/1e6,&result);return result;}
static napi_value registerModule(napi_env e,napi_value exports){napi_property_descriptor props[]={{"now",nullptr,monotonic,nullptr,nullptr,nullptr,napi_default,nullptr},{"configure",nullptr,init,nullptr,nullptr,nullptr,napi_default,nullptr},{"fill",nullptr,fill,nullptr,nullptr,nullptr,napi_default,nullptr}};napi_define_properties(e,exports,3,props);return exports;}
static napi_module mod={1,0,nullptr,registerModule,"batchprep",nullptr,{0}};
extern "C" __attribute__((constructor)) void RegisterBatchPrep(){napi_module_register(&mod);}
