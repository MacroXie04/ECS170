// Explicit C API handles keep training, evaluation and saved models independent.
#include <mindspore/model.h>
#include <mindspore/context.h>
#include <mindspore/tensor.h>
#include <vector>
#include <string>
#include <fstream>
#include <set>
static std::vector<OH_AI_ModelHandle> models;
static std::vector<std::vector<uint8_t>> modelBuffers;
static std::vector<OH_AI_TensorHandle> tensors;
static napi_value number(napi_env e,double n){napi_value v;napi_create_double(e,n,&v);return v;}
static napi_value textValue(napi_env e,const char *s){napi_value v;napi_create_string_utf8(e,s,NAPI_AUTO_LENGTH,&v);return v;}
static std::string textArg(napi_env e,napi_value v){size_t n=0;napi_get_value_string_utf8(e,v,nullptr,0,&n);std::string s(n+1,'\0');napi_get_value_string_utf8(e,v,s.data(),n+1,&n);s.resize(n);return s;}
#ifdef ENABLE_MNN_PROBE
#include "mnn_probe.inc"
#endif
static napi_value mnnAsync(napi_env e,napi_callback_info info){
#ifdef ENABLE_MNN_PROBE
 return mnnprobe::asyncCall(e,info);
#else
 napi_throw_error(e,nullptr,"MNN probe not compiled; use harmonyos/mnn/build_probe.sh");return nullptr;
#endif
}
static napi_value nativeCall(napi_env e,napi_callback_info info){
 napi_value a[4];size_t n=4;napi_get_cb_info(e,info,&n,a,nullptr,nullptr);std::string op=textArg(e,a[0]);
 auto fail=[&](const std::string&s)->napi_value{napi_throw_error(e,nullptr,s.c_str());return nullptr;};
 if(op=="backendInfo"){
  size_t count=0;auto descs=OH_AI_GetAllNNRTDeviceDescs(&count);std::string result="NNRT device count="+std::to_string(count);
  for(size_t i=0;descs&&i<count;i++){auto d=OH_AI_GetElementOfNNRTDeviceDescs(descs,i);auto name=OH_AI_GetNameFromNNRTDeviceDesc(d);result+="\nname="+std::string(name?name:"")+" type="+std::to_string(OH_AI_GetTypeFromNNRTDeviceDesc(d))+" id="+std::to_string(OH_AI_GetDeviceIdFromNNRTDeviceDesc(d));}
  if(descs)OH_AI_DestroyAllNNRTDeviceDescs(&descs);
  return textValue(e,result.c_str());
 }
 if(op=="backendLibraries"){
  std::ifstream maps("/proc/self/maps");std::string line,result;std::set<std::string> paths;
  while(std::getline(maps,line)){auto pos=line.find('/');if(pos==std::string::npos)continue;auto path=line.substr(pos);if(path.find("mindspore")!=std::string::npos||path.find("OpenCL")!=std::string::npos||path.find("vulkan")!=std::string::npos||path.find("nnrt")!=std::string::npos)paths.insert(path);}
  for(const auto &path:paths)result+=path+"\n";return textValue(e,result.c_str());
 }
 if(op=="load"){
  void *data;size_t bytes;if(!buf(e,a[1],&data,&bytes))return fail("Model buffer required");bool train=false;napi_get_value_bool(e,a[2],&train);
  const std::string backend=n>3?textArg(e,a[3]):"cpu";
  if(backend!="cpu"&&backend!="gpu"&&backend!="gpu_cpu"&&backend!="nnrt_gpu"&&backend!="nnrt_gpu_cpu"&&backend!="nnrt"&&backend!="nnrt_cpu")return fail("Unknown backend: "+backend);
  auto context=OH_AI_ContextCreate();OH_AI_ContextSetThreadNum(context,4);
  if(backend!="cpu"){
   OH_AI_DeviceInfoHandle accelerator=nullptr;
   if(backend=="gpu"||backend=="gpu_cpu")accelerator=OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_GPU);
   else if(backend=="nnrt_gpu"||backend=="nnrt_gpu_cpu")accelerator=OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_GPU);
   else accelerator=OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_NNRT);
   if(!accelerator){OH_AI_ContextDestroy(&context);return fail("Requested backend unavailable: "+backend);}
   if(backend=="gpu"||backend=="gpu_cpu")OH_AI_DeviceInfoSetEnableFP16(accelerator,false);
   OH_AI_ContextAddDeviceInfo(context,accelerator);
  }
  if(backend=="cpu"||backend=="gpu_cpu"||backend=="nnrt_gpu_cpu"||backend=="nnrt_cpu"){auto cpu=OH_AI_DeviceInfoCreate(OH_AI_DEVICETYPE_CPU);OH_AI_DeviceInfoSetEnableFP16(cpu,false);OH_AI_ContextAddDeviceInfo(context,cpu);}
  modelBuffers.emplace_back(static_cast<uint8_t*>(data),static_cast<uint8_t*>(data)+bytes);data=modelBuffers.back().data();auto model=OH_AI_ModelCreate();
  OH_AI_Status status;
  // Build transfers configuration ownership to the model; destroying it here double-frees it.
  if(train){auto cfg=OH_AI_TrainCfgCreate();const char *loss[]={"loss"};OH_AI_TrainCfgSetLossName(cfg,loss,1);OH_AI_TrainCfgSetOptimizationLevel(cfg,static_cast<OH_AI_OptimizationLevel>(0));status=OH_AI_TrainModelBuild(model,data,bytes,OH_AI_MODELTYPE_MINDIR,context,cfg);}else status=OH_AI_ModelBuild(model,data,bytes,OH_AI_MODELTYPE_MINDIR,context);
  if(status!=0){OH_AI_ModelDestroy(&model);modelBuffers.pop_back();return fail("Native model build failed ("+backend+"): "+std::to_string(static_cast<int32_t>(status)));}models.push_back(model);return number(e,models.size()-1);
 }
 uint32_t id=integer(e,a[1]);
 if(op=="data"||op=="setData"){
  if(id>=tensors.size())return fail("Invalid tensor handle");auto t=tensors[id];size_t bytes=OH_AI_TensorGetDataSize(t);
  if(op=="data"){napi_value out;void *p;napi_create_arraybuffer(e,bytes,&p,&out);auto source=OH_AI_TensorGetData(t);if(!source)return fail("Null tensor data");memcpy(p,source,bytes);return out;}
  void *p;size_t size;if(!buf(e,a[2],&p,&size)||size!=bytes)return fail("Tensor byte count mismatch");memcpy(OH_AI_TensorGetMutableData(t),p,size);return number(e,0);
 }
 if(id>=models.size())return fail("Invalid model handle");auto m=models[id];OH_AI_Status status=OH_AI_STATUS_SUCCESS;
 if(op=="inputs"||op=="weights"||op=="predict"){
  OH_AI_TensorHandleArray ts{};
  if(op=="inputs")ts=OH_AI_ModelGetInputs(m);else if(op=="weights")ts=OH_AI_ModelGetWeights(m);else{status=OH_AI_ModelPredict(m,OH_AI_ModelGetInputs(m),&ts,nullptr,nullptr);if(status!=0)return fail("Predict failed: "+std::to_string(status));}
  napi_value out;napi_create_array_with_length(e,ts.handle_num,&out);
  for(size_t i=0;i<ts.handle_num;i++){auto t=ts.handle_list[i];tensors.push_back(t);napi_value obj;napi_create_object(e,&obj);napi_set_named_property(e,obj,"handle",number(e,tensors.size()-1));napi_set_named_property(e,obj,"name",textValue(e,OH_AI_TensorGetName(t)));napi_set_named_property(e,obj,"elementNum",number(e,OH_AI_TensorGetElementNum(t)));size_t dims;auto shape=OH_AI_TensorGetShape(t,&dims);napi_value arr;napi_create_array_with_length(e,dims,&arr);for(size_t j=0;j<dims;j++)napi_set_element(e,arr,j,number(e,shape[j]));napi_set_named_property(e,obj,"shape",arr);napi_set_element(e,out,i,obj);}return out;
 }
 if(op=="destroy"){OH_AI_ModelDestroy(&models[id]);modelBuffers[id].clear();modelBuffers[id].shrink_to_fit();return number(e,0);}
 if(op=="step"){
  auto weights=OH_AI_ModelGetWeights(m);std::vector<OH_AI_TensorHandle> stats;std::vector<std::vector<float>> prior;
  for(size_t i=0;i<weights.handle_num;i++){auto t=weights.handle_list[i];std::string name=OH_AI_TensorGetName(t);if(name=="moving_mean"||name=="moving_variance"||name.find(".moving_mean")!=std::string::npos||name.find(".moving_variance")!=std::string::npos){auto p=static_cast<const float*>(OH_AI_TensorGetData(t));stats.push_back(t);prior.emplace_back(p,p+OH_AI_TensorGetElementNum(t));}}
  status=OH_AI_RunStep(m,nullptr,nullptr);
  if(status==0)for(size_t i=0;i<stats.size();i++){auto p=static_cast<float*>(OH_AI_TensorGetMutableData(stats[i]));for(size_t j=0;j<prior[i].size();j++)p[j]=0.9f*prior[i][j]+0.1f*p[j];}
 }
 else if(op=="lr"){if(n>2){double lr;napi_get_value_double(e,a[2],&lr);status=OH_AI_ModelSetLearningRate(m,lr);}else return number(e,OH_AI_ModelGetLearningRate(m));}
 else if(op=="mode"){if(n>2){bool mode;napi_get_value_bool(e,a[2],&mode);status=OH_AI_ModelSetTrainMode(m,mode);}else{napi_value v;napi_get_boolean(e,OH_AI_ModelGetTrainMode(m),&v);return v;}}
 else if(op=="update"){uint32_t count;napi_get_array_length(e,a[2],&count);std::vector<OH_AI_TensorHandle> ws;for(uint32_t i=0;i<count;i++){napi_value v;napi_get_element(e,a[2],i,&v);auto tid=integer(e,v);if(tid>=tensors.size())return fail("Invalid update tensor");ws.push_back(tensors[tid]);}auto target=OH_AI_ModelGetWeights(m);for(auto source:ws){bool found=false;for(size_t j=0;j<target.handle_num;j++){auto dest=target.handle_list[j];if(strcmp(OH_AI_TensorGetName(source),OH_AI_TensorGetName(dest))==0){size_t bytes=OH_AI_TensorGetDataSize(source);if(bytes!=OH_AI_TensorGetDataSize(dest))return fail("Weight size mismatch");memcpy(OH_AI_TensorGetMutableData(dest),OH_AI_TensorGetData(source),bytes);found=true;break;}}if(!found)return fail("Missing target weight");}}
 else if(op=="export"){std::string path=textArg(e,a[2]);bool infer;napi_get_value_bool(e,a[3],&infer);status=OH_AI_ExportModel(m,OH_AI_MODELTYPE_MINDIR,path.c_str(),static_cast<OH_AI_QuantizationType>(0),infer,nullptr,0);}
 else return fail("Unknown native operation: "+op);
 if(status!=0)return fail(op+" failed: "+std::to_string(status));return number(e,0);
}
