######################################################################################
#  Method A (Python C-API) create a module that import like a normal Python library. #
#                      Loading the compiled fast_math.so library                     #
#                           calling the C function directly                          #
######################################################################################
import sys
import time
import platform
import multiprocessing

######################################################################################
#                                                                                    #
#           Collects system and hardware information for benchmark context.          #
#                                                                                    #
######################################################################################
try:
    import psutil
except ImportError:
    psutil = None

def get_hardware_info():
    info = []
    info.append("-" * 65)
    info.append("RUNNING ON ENVIRONMENT:")
    
    # 1. OS and Python Info
    info.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    info.append(f"Python Version: {sys.version.split()[0]}")
    
    # 2. CPU Info
    # platform.processor() often returns 'amdk8' or empty on some Linux distros
    # We use multiprocessing for a reliable count
    logical_cores = multiprocessing.cpu_count()
    info.append(f"CPU Cores: {logical_cores} (Logical)")
    
    # 3. RAM Info (Requires psutil)
    if psutil:
        total_ram = round(psutil.virtual_memory().total / (1024**3), 2)
        info.append(f"Total RAM: {total_ram} GB")
    else:
        info.append("Total RAM: [psutil not installed]")
    return "\n".join(info)
  
######################################################################################
#  Method A (Python C-API) create a module that import like a normal Python library. #
#                      Loading the compiled fast_math.so library                     #
#                           calling the C function directly                          #
######################################################################################  
def c_bridge(count):
  # Ensure Python can find the compiled module fast_math.so file
  # sys.path.append('path') or python search in app folder
  hw_header = get_hardware_info()
  benchmark_output = "-" * 65 + "\n---    Performance Benchmark: C Extension vs. Pure Python     ---\n" + "-" * 65
  time_c = 0
  time_py = 0
  try:
      # import compiled module like a normal library
      import fast_math
      benchmark_output += "\nSUCCESS: fast_math module imported correctly!"
            
      # Create a large dataset (10 Million items)  count = 10_000_000 
      data = list(range(count))
      benchmark_output += f"\nDataset size: {count:,} integers\n"
      
      # --- TEST 1: C-EXTENSION ---
      start_c = time.perf_counter()
      result_c = fast_math.fast_sum(data) # fast_math - C module
      end_c = time.perf_counter()         # fast_sum  - function in C module
      time_c = end_c - start_c
      
      # --- TEST 2: PURE PYTHON LOOP ---
      start_py = time.perf_counter()
      sum_py = 0
      for n in data:
          sum_py += n
      end_py = time.perf_counter()
      time_py = end_py - start_py
      
      # --- TEST 3: PYTHON BUILT-IN SUM (Which is also written in C) ---
      start_builtin = time.perf_counter()
      sum_builtin = sum(data)
      end_builtin = time.perf_counter()
      time_builtin = end_builtin - start_builtin
      
      # Display Results
      benchmark_output += f"\n{'Method':<25} | {'Result':<20} | {'Execution Time':<15}\n"
      benchmark_output += "-" * 65
      benchmark_output += f"\n{'C-Extension (Custom)':<25} | {result_c:<20} | {time_c:.6f} s"
      benchmark_output += f"\n{'Python Manual Loop':<25} | {sum_py:<20} | {time_py:.6f} s"
      benchmark_output += f"\n{'Python Built-in sum()':<25} | {sum_builtin:<20} | {time_builtin:.6f} s\n"
      benchmark_output += "-" * 65
      if time_c > 0:
          speedup = time_py / time_c
          benchmark_output += f"\nConclusion: C-Extension is ~ {speedup:.0f}x faster than a manual Python loop."
  except ImportError as e:
      benchmark_output += f"\nERROR: Could not find fast_math. {e}"
      # print(f"Current search paths: {sys.path}")
  return f"{hw_header}\n{benchmark_output}", time_c, time_py

# # Method B (ctypes) is better for wrapping existing C libraries 
# # (like OpenSSL or SQLite) without changing their code.
# import ctypes
# lib = ctypes.CDLL('/home/dr/c/fast_math.so')                       # Load the library
# lib.c_sum.argtypes = [ctypes.POINTER(ctypes.c_long), ctypes.c_int] # Define input types for safety
# lib.c_sum.restype = ctypes.c_long
# data = (ctypes.c_long * 5)(10, 20, 30, 40, 50)                     # Prepare data
# result = lib.c_sum(data, 5)                                        # Call
# print(f"Result via ctypes: {result}")



