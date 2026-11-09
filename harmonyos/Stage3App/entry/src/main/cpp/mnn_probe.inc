// Optional MNN training probe. Normal Stage 3 runs keep their existing backend.
#include <dlfcn.h>
#include <vulkan/vulkan.h>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

#include "ADAM.hpp"
#include "Initializer.hpp"
#include "NN.hpp"
#include "OpGrad.hpp"
#include "RandomGenerator.hpp"
#include "core/TensorUtils.hpp"
namespace mnnprobe {
using namespace MNN;
using namespace MNN::Express;
static double now() {
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static std::string quote(const std::string& s) {
    std::ostringstream o;
    o << '"';
    for (char c : s) {
        if (c == '"' || c == '\\')
            o << '\\' << c;
        else if (c == '\n')
            o << "\\n";
        else if (static_cast<unsigned char>(c) >= 32)
            o << c;
    }
    o << '"';
    return o.str();
}
template <class T>
static std::vector<T> read(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot read " + path);
    auto bytes = f.tellg();
    if (bytes < 0 || size_t(bytes) % sizeof(T)) throw std::runtime_error("Bad file size");
    std::vector<T> v(size_t(bytes) / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    if (!f) throw std::runtime_error("Incomplete read");
    return v;
}
static std::string vulkanDevices() {
    // MNN's static Vulkan wrapper defines same-named function-pointer symbols.
    // Resolve the public loader explicitly instead of calling those data symbols.
    void* library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) return "Vulkan loader unavailable";
    auto create = reinterpret_cast<PFN_vkCreateInstance>(dlsym(library, "vkCreateInstance"));
    auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(dlsym(library, "vkEnumeratePhysicalDevices"));
    auto properties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(dlsym(library, "vkGetPhysicalDeviceProperties"));
    auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(dlsym(library, "vkDestroyInstance"));
    if (!create || !enumerate || !properties || !destroy) {
        dlclose(library);
        return "Vulkan loader entry points unavailable";
    }
    VkApplicationInfo a{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    a.pApplicationName = "Stage3TrainingProbe";
    a.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &a;
    VkInstance instance;
    auto status = create(&ci, nullptr, &instance);
    if (status != VK_SUCCESS) {
        dlclose(library);
        return "vkCreateInstance=" + std::to_string(status);
    }
    uint32_t n = 0;
    enumerate(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> ds(n);
    enumerate(instance, &n, ds.data());
    std::string out;
    for (auto d : ds) {
        VkPhysicalDeviceProperties p{};
        properties(d, &p);
        out += std::string(p.deviceName) + " vendor=" + std::to_string(p.vendorID) +
               " api=" + std::to_string(p.apiVersion) + "\n";
    }
    destroy(instance, nullptr);
    dlclose(library);
    return out;
}
class MnistNet final : public Module {
    std::shared_ptr<Module> c1, b1, c2, b2, f1, drop, f2;

   public:
    MnistNet() {
        NN::ConvOption c;
        c.channel = {1, 32};
        c.kernelSize = {5, 5};
        c.padMode = CAFFE;
        c.pads = {2, 2};
        c1.reset(NN::Conv(c, false, std::shared_ptr<Initializer>(Initializer::MSRA(Initializer::FANOUT))));
        b1.reset(NN::BatchNorm(32, 4, .9f, 1e-5f));
        c.channel = {32, 64};
        c2.reset(NN::Conv(c, false, std::shared_ptr<Initializer>(Initializer::MSRA(Initializer::FANOUT))));
        b2.reset(NN::BatchNorm(64, 4, .9f, 1e-5f));
        f1.reset(NN::Linear(64 * 7 * 7, 128));
        drop.reset(NN::Dropout(.25));
        f2.reset(NN::Linear(128, 10));
        registerModel({c1, b1, c2, b2, f1, drop, f2});
    }
    std::vector<VARP> onForward(const std::vector<VARP>& ins) override {
        auto x = _Convert(ins[0], NC4HW4);
        x = _MaxPool(_Relu(b1->forward(c1->forward(x))), {2, 2}, {2, 2});
        x = _MaxPool(_Relu(b2->forward(c2->forward(x))), {2, 2}, {2, 2});
        x = _Reshape(_Convert(x, NCHW), {0, -1});
        return {f2->forward(drop->forward(_Relu(f1->forward(x))))};
    }
};
struct State {
    std::string dir, id, backend, status = "starting", error, hardware, reloadedFrom, phase = "init";
    int steps = 0;
    double start = now(), reloadError = -1, firstPassLoss = -1, lastPassLoss = -1;
    size_t changed = 0;
    std::shared_ptr<MnistNet> model;
    std::unique_ptr<MNN::Train::ADAM> optimizer;
    std::shared_ptr<Executor> executor;
    VARP x, y;
    std::vector<float> images, labels, before, losses, stepTimes, gpuTimes;
    std::vector<int> learnedPositions;
    std::map<std::string, size_t> trace;
    std::map<std::string, size_t> learnedTensorUpdates;
    std::string report() {
        std::ostringstream o;
        o << std::setprecision(10)
          << "{\"framework\":\"MNN\",\"model\":\"MNIST kernel5 architecture port\",\"id\":" << quote(id)
          << ",\"requestedBackend\":" << quote(backend) << ",\"state\":" << quote(status)
          << ",\"error\":" << quote(error) << ",\"reloadedFrom\":" << quote(reloadedFrom)
          << ",\"vulkanDevices\":" << quote(hardware) << ",\"steps\":" << steps
          << ",\"cpuThreads\":4,\"batchSize\":128,\"availableTrainingImages\":1024,\"uniqueTrainingImages\":"
          << std::min(steps, 8) * 128 << ",\"trainingExamplesSeen\":" << steps * 128
          << ",\"changedLearnedWeights\":" << changed << ",\"reloadMaxLogitError\":" << reloadError
          << ",\"gpuTimestampProfiling\":" << (MNN_PROBE_GPU_PROFILE ? "true" : "false")
          << ",\"firstPassMeanLoss\":" << firstPassLoss << ",\"lastPassMeanLoss\":" << lastPassLoss
          << ",\"elapsedMs\":" << now() - start;
        auto arr = [&](const char* key, const std::vector<float>& a) {
            o << ",\"" << key << "\":[";
            for (size_t i = 0; i < a.size(); i++) {
                if (i) o << ',';
                if (std::isfinite(a[i]))
                    o << a[i];
                else
                    o << "null";
            }
            o << ']';
        };
        arr("loss", losses);
        arr("stepMs", stepTimes);
        arr("lastGpuMs", gpuTimes);
        o << ",\"operatorOutputBackendCounts\":{";
        bool first = true;
        for (auto& t : trace) {
            if (!first) o << ',';
            first = false;
            o << quote(t.first) << ':' << t.second;
        }
        o << "},\"learnedTensorUpdates\":{";
        first = true;
        for (auto& t : learnedTensorUpdates) {
            if (!first) o << ',';
            first = false;
            o << quote(t.first) << ':' << t.second;
        }
        o << "}}";
        return o.str();
    }
    void save() { std::ofstream(dir + "/" + id + "_mnn.json") << report(); }
    std::vector<float> weights() {
        std::vector<float> out;
        auto params = model->parameters();
        for (int i : learnedPositions) {
            auto p = params[i];
            auto info = p->getInfo();
            auto v = p->readMap<float>();
            if (!info || !v) throw std::runtime_error("Cannot read learned weight");
            out.insert(out.end(), v, v + info->size);
        }
        return out;
    }
    void saveFloats(const std::string& suffix, const std::vector<float>& values) {
        std::ofstream f(dir + "/" + id + suffix, std::ios::binary);
        f.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        if (!f) throw std::runtime_error("Cannot save numerical check");
    }
    void checkGradient(VARP loss) {
        phase = "first_step_gradient_check";
        auto params = model->parameters();
        std::set<VARP> learned;
        for (auto index : learnedPositions) learned.insert(params[index]);
        auto gradients = MNN::OpGrad::grad(loss, learned);
        std::vector<VARP> expressions;
        for (auto index : learnedPositions) {
            if (!gradients.count(params[index])) throw std::runtime_error("Missing learned gradient");
            expressions.push_back(gradients[params[index]]);
        }
        Variable::prepareCompute(expressions);
        std::vector<float> values;
        for (auto gradient : expressions) {
            auto info = gradient->getInfo();
            auto ptr = gradient->readMap<float>();
            if (!ptr || !info) throw std::runtime_error("Cannot read first-step gradient");
            for (int i = 0; i < info->size; i++) {
                if (!std::isfinite(ptr[i])) throw std::runtime_error("Nonfinite first-step gradient");
                values.push_back(ptr[i]);
            }
        }
        if (values.size() != before.size()) throw std::runtime_error("Gradient size mismatch");
        saveFloats("_step1_gradients.bin", values);
    }
    void batch(int index) {
        memcpy(x->writeMap<float>(), images.data() + size_t(index) * 128 * 784, 128 * 784 * sizeof(float));
        memcpy(y->writeMap<float>(), labels.data() + size_t(index) * 128 * 10, 128 * 10 * sizeof(float));
    }
    void init() {
        hardware = vulkanDevices();
        status = "creating_runtime";
        save();
        MNNForwardType type = backend.rfind("opencl", 0) == 0 ? MNN_FORWARD_OPENCL
                              : backend.rfind("vulkan", 0) == 0 ? MNN_FORWARD_VULKAN
                                                              : MNN_FORWARD_CPU;
        BackendConfig config;
        config.precision = BackendConfig::Precision_High;
        executor = Executor::getGlobalExecutor();
        executor->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
        int mode = backend == "opencl_buffer" ? MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER : 4;
        if (backend == "vulkan_batch") mode |= MNN_GPU_RECORD_BATCH;
        if (type != MNN_FORWARD_CPU) executor->setGlobalExecutorConfig(type, config, mode);
        auto rt = Executor::getRuntime();
        if (!rt.first.count(type)) throw std::runtime_error("Requested MNN runtime unavailable; CPU fallback rejected");
        executor->setCallBack([](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
                              [this](const std::vector<Tensor*>& ts, const OperatorInfo* op) {
                                  std::set<int> types;
                                  for (auto t : ts) {
                                      auto b = TensorUtils::getDescribeOrigin(t)->getBackend();
                                      types.insert(b ? int(b->type()) : -1);
                                  }
                                  for (auto type : types)
                                      trace[phase + "/" + op->type() + "/" + std::to_string(type)]++;
                                  return true;
                              });
        images = read<float>(dir + "/mnn_mnist_images.bin");
        labels = read<float>(dir + "/mnn_mnist_labels.bin");
        if (images.size() != 1024 * 784 || labels.size() != 1024 * 10)
            throw std::runtime_error("Expected 1024 real MNIST examples");
        RandomGenerator::generator().seed(2026);
        model = std::make_shared<MnistNet>();
        model->setIsTraining(true);
        auto params = model->parameters();
        for (size_t i = 0; i < params.size(); i++)
            if (params[i]->expr().first->inputType() == VARP::TRAINABLE) learnedPositions.push_back(i);
        before = weights();
        if (before.empty()) throw std::runtime_error("No learned parameters found");
        saveFloats("_initial_weights.bin", before);
        optimizer.reset(new MNN::Train::ADAM(model));
        optimizer->setLearningRate(.001f);
        optimizer->setMomentum(.9f);
        optimizer->setMomentum2(.999f);
        optimizer->setWeightDecay(0);
        x = _Input({128, 1, 28, 28}, NCHW);
        x->setName("images");
        y = _Input({128, 10}, NCHW);
        status = "ready";
        save();
    }
    void step() {
        auto started = now();
        phase = "input";
        batch(steps % 8);
        phase = "forward";
        auto logits = model->forward(x);
        auto max = _ReduceMax(logits, {1}, true);
        auto logz = _Log(_ReduceSum(_Exp(logits - max), {1}, true)) + max;
        auto loss = _ReduceMean(_ReduceSum(y * (logz - logits), {1}), {});
        auto value = loss->readMap<float>();
        if (!value || !std::isfinite(value[0])) throw std::runtime_error("Nonfinite or missing loss");
        losses.push_back(value[0]);
        gpuTimes.push_back(executor->getLastGpuTimeMs());
        if (steps == 0) checkGradient(loss);
        phase = "backward_optimizer";
        if (!optimizer->step(loss)) throw std::runtime_error("Optimizer failed");
        gpuTimes.push_back(executor->getLastGpuTimeMs());
        steps++;
        stepTimes.push_back(now() - started);
        if (steps == 1) saveFloats("_step1_weights.bin", weights());
        status = "running";
        save();
    }
    void finish() {
        if (steps >= 16) {
            firstPassLoss = 0;
            lastPassLoss = 0;
            for (int i = 0; i < 8; i++) {
                firstPassLoss += losses[i] / 8.0;
                lastPassLoss += losses[losses.size() - 8 + i] / 8.0;
            }
            if (lastPassLoss >= firstPassLoss) throw std::runtime_error("Mean training loss did not decrease");
        }
        phase = "checkpoint";
        auto after = weights();
        if (after.size() != before.size()) throw std::runtime_error("Weight count changed");
        for (size_t i = 0; i < after.size(); i++) {
            if (!std::isfinite(after[i])) throw std::runtime_error("Nonfinite weight");
            changed += after[i] != before[i];
        }
        if (changed == 0) throw std::runtime_error("No learned weight updated");
        size_t offset = 0;
        auto parameters = model->parameters();
        for (auto pos : learnedPositions) {
            auto info = parameters[pos]->getInfo();
            std::string key = std::to_string(pos) + ":";
            for (auto dim : info->dim) key += std::to_string(dim) + "x";
            size_t count = 0;
            for (int i = 0; i < info->size; i++) count += after[offset + i] != before[offset + i];
            learnedTensorUpdates[key] = count;
            offset += info->size;
        }
        model->setIsTraining(false);
        batch(0);
        auto prediction = model->forward(x);
        prediction->setName("logits");
        auto ptr = prediction->readMap<float>();
        if (!ptr) throw std::runtime_error("Inference failed");
        std::vector<float> expected(ptr, ptr + 1280);
        Variable::save(model->parameters(), (dir + "/" + id + "_params.mnn").c_str());
        Variable::save({prediction}, (dir + "/" + id + "_model.mnn").c_str());
        std::ofstream f(dir + "/" + id + "_expected.bin", std::ios::binary);
        f.write(reinterpret_cast<char*>(expected.data()), expected.size() * 4);
        f.close();
        phase = "reload";
        auto saved = Variable::loadMap((dir + "/" + id + "_model.mnn").c_str());
        if (!saved.count("images") || !saved.count("logits")) throw std::runtime_error("Saved graph endpoints missing");
        memcpy(saved["images"]->writeMap<float>(), images.data(), 128 * 784 * 4);
        auto actual = saved["logits"]->readMap<float>();
        if (!actual) throw std::runtime_error("Reload failed");
        reloadError = 0;
        for (int i = 0; i < 1280; i++) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
                throw std::runtime_error("Nonfinite reload output or reference");
            reloadError = std::max(reloadError, double(std::abs(expected[i] - actual[i])));
            if (std::abs(expected[i] - actual[i]) > 1e-4f + 1e-4f * std::abs(expected[i]))
                throw std::runtime_error("Reload output does not match saved model reference");
        }
        status = "completed";
        save();
    }
    void reload(const std::string& source) {
        phase = "independent_process_reload";
        reloadedFrom = source;
        auto expected = read<float>(dir + "/" + source + "_expected.bin");
        if (expected.size() != 1280) throw std::runtime_error("Invalid saved reference");
        auto saved = Variable::loadMap((dir + "/" + source + "_model.mnn").c_str());
        if (!saved.count("images") || !saved.count("logits")) throw std::runtime_error("Saved graph endpoints missing");
        memcpy(saved["images"]->writeMap<float>(), images.data(), 128 * 784 * 4);
        auto actual = saved["logits"]->readMap<float>();
        if (!actual) throw std::runtime_error("Independent reload failed");
        reloadError = 0;
        for (int i = 0; i < 1280; i++) {
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
                throw std::runtime_error("Nonfinite reload output or reference");
            reloadError = std::max(reloadError, double(std::abs(expected[i] - actual[i])));
            if (std::abs(expected[i] - actual[i]) > 1e-4f + 1e-4f * std::abs(expected[i]))
                throw std::runtime_error("Independent reload output does not match saved model reference");
        }
        status = "completed";
        save();
    }
};
static std::unique_ptr<State>& state() {
    static std::unique_ptr<State> p;
    return p;
}

static std::string dispatch(const std::string& op, const std::string& a, const std::string& b, const std::string& c) {
    auto& s = state();
    try {
        if (op == "mnn_init") {
            if (s) throw std::runtime_error("Use a fresh app process for each MNN run");
            s.reset(new State());
            s->dir = a;
            s->backend = b;
            s->id = c;
            s->save();
            s->init();
        } else {
            if (!s) throw std::runtime_error("MNN probe not initialized");
            if (op == "mnn_step")
                s->step();
            else if (op == "mnn_finish")
                s->finish();
            else if (op == "mnn_reload")
                s->reload(a);
            else
                throw std::runtime_error("Unknown MNN probe operation");
        }
    } catch (const std::exception& ex) {
        if (!s) throw;
        s->status = "error";
        s->error = ex.what();
        s->save();
    }
    return s->report();
}
struct AsyncCall {
    napi_async_work work = nullptr;
    napi_deferred deferred = nullptr;
    std::string op, args[3], result, error;
};
static std::atomic<bool>& busy() {
    static std::atomic<bool> value{false};
    return value;
}
static napi_value asyncCall(napi_env env, napi_callback_info info) {
    napi_value args[4];
    size_t count = 4;
    napi_get_cb_info(env, info, &count, args, nullptr, nullptr);
    if (count < 1 || busy().exchange(true)) {
        napi_throw_error(env, nullptr, "MNN requires serialized calls");
        return nullptr;
    }
    auto call = new AsyncCall();
    call->op = textArg(env, args[0]);
    for (size_t i = 1; i < count; i++) call->args[i - 1] = textArg(env, args[i]);
    napi_value promise, name;
    napi_create_promise(env, &call->deferred, &promise);
    napi_create_string_utf8(env, "Stage3MNN", NAPI_AUTO_LENGTH, &name);
    auto created = napi_create_async_work(
        env, nullptr, name,
        [](napi_env, void* data) {
            auto c = static_cast<AsyncCall*>(data);
            try {
                c->result = dispatch(c->op, c->args[0], c->args[1], c->args[2]);
            } catch (const std::exception& ex) {
                c->error = ex.what();
            }
        },
        [](napi_env env, napi_status status, void* data) {
            auto c = static_cast<AsyncCall*>(data);
            busy().store(false);
            if (status != napi_ok || !c->error.empty()) {
                auto text = textValue(env, c->error.empty() ? "MNN async execution failed" : c->error.c_str());
                napi_value error;
                napi_create_error(env, nullptr, text, &error);
                napi_reject_deferred(env, c->deferred, error);
            } else
                napi_resolve_deferred(env, c->deferred, textValue(env, c->result.c_str()));
            napi_delete_async_work(env, c->work);
            delete c;
        },
        call, &call->work);
    if (created != napi_ok || napi_queue_async_work(env, call->work) != napi_ok) {
        busy().store(false);
        napi_value error;
        napi_create_error(env, nullptr, textValue(env, "Cannot queue MNN work"), &error);
        napi_reject_deferred(env, call->deferred, error);
        if (call->work) napi_delete_async_work(env, call->work);
        delete call;
    }
    return promise;
}
}  // namespace mnnprobe
