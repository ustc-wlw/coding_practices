<mark>**nvprof性能调试工具**</mark>

SM上线程束占用率

nvprof --metrics  **achieved_occupancy**  user_app  app_params_list

查内核的内存读取效率

nvprof --metrics **gld_throughput**  user_app app_params_list

检测全局加载效率

nvprof --metrics  **gld_efficiency**  user_app app_params_list

每个线程束上执行指令数量的平均值

nvprof --metrics   **inst_per_warp**   user_app app_params_list

设备内存读取吞吐量指标

nvprof --metrics  **dram_read_throughput**   user_app app_params_list

查看线程束同步阻塞

nvprof --metrics **stall_sync** user_app app_params_list


