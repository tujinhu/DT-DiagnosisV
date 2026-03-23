
import os
import sys
import time
import threading
import subprocess
from datetime import datetime
from collections import defaultdict

import psutil
import pandas as pd
import numpy as np


EXTERNAL_PROCESSES = [
    'QGroundControl.exe',   
    'RflySim3D.exe',        
    'CopterSim.exe',        
]

# GPU monitoring - try to import pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVIDIA = True
    print("[PerformanceMonitor] NVIDIA GPU monitoring enabled")
except Exception as e:
    HAS_NVIDIA = False
    print(f"[PerformanceMonitor] Warning: pynvml not available ({e}), using nvidia-smi fallback")


def get_gpu_info_nvidia_smi():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def find_processes_by_name(name):
    processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] and name.lower() in proc.info['name'].lower():
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes


class PerformanceMonitor:

    
    def __init__(self, target_pid=None, output_dir=None, sample_interval=0.1):

        self.target_pid = target_pid or os.getpid()
        self.output_dir = output_dir or sys.path[0]
        self.sample_interval = sample_interval
        

        self.system_data = []  
        self.model_data = []   
        

        try:
            self.process = psutil.Process(self.target_pid)
            self.process_name = self.process.name()
        except psutil.NoSuchProcess:
            self.process = None
            self.process_name = "Unknown"
            print(f"[PerformanceMonitor] Warning: Process {self.target_pid} not found")
        

        self.gpu_handle = None
        self.gpu_name = "Unknown"
        if HAS_NVIDIA:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                if isinstance(self.gpu_name, bytes):
                    self.gpu_name = self.gpu_name.decode('utf-8')
            except Exception as e:
                print(f"[PerformanceMonitor] Warning: Failed to get GPU handle: {e}")
        

        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()
        

        self.start_time = None
        

        self.external_processes_data = [] 
        self.external_pids = {}  # {process_name: [pid, ...]}
        
    def _get_cpu_usage(self):
        if self.process is None:
            return 0.0
        try:
            return self.process.cpu_percent(interval=None)
        except Exception:
            return 0.0
    
    def _get_memory_usage(self):
        if self.process is None:
            return 0.0
        try:
            mem_info = self.process.memory_info()
            return mem_info.rss / (1024 * 1024)  # 转换为MB
        except Exception:
            return 0.0
    
    def _get_system_cpu_usage(self):
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
    
    def _get_system_memory_usage(self):
        try:
            mem = psutil.virtual_memory()
            return mem.used / (1024 * 1024), mem.percent
        except Exception:
            return 0.0, 0.0
    
    def _get_gpu_usage(self):
        if HAS_NVIDIA and self.gpu_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = util.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_used = mem_info.used / (1024 * 1024)
                gpu_mem_total = mem_info.total / (1024 * 1024)
                return gpu_util, gpu_mem_used, gpu_mem_total
            except Exception:
                pass
        
        # Fallback to nvidia-smi
        return get_gpu_info_nvidia_smi()
    
    def _get_thread_count(self):
        if self.process is None:
            return 0
        try:
            return self.process.num_threads()
        except Exception:
            return 0
    
    def _monitor_loop(self):
        if self.process:
            try:
                self.process.cpu_percent(interval=None)
            except Exception:
                pass
        psutil.cpu_percent(interval=None)
        
        while self._running:
            timestamp = time.time()
            elapsed = timestamp - self.start_time
            
            proc_cpu = self._get_cpu_usage()
            proc_mem = self._get_memory_usage()
            sys_cpu = self._get_system_cpu_usage()
            sys_mem_used, sys_mem_percent = self._get_system_memory_usage()
            gpu_util, gpu_mem_used, gpu_mem_total = self._get_gpu_usage()
            thread_count = self._get_thread_count()
            
            record = {
                'timestamp': timestamp,
                'elapsed_sec': round(elapsed, 3),
                'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'process_cpu_percent': round(proc_cpu, 2),
                'process_memory_mb': round(proc_mem, 2),
                'process_threads': thread_count,
                'system_cpu_percent': round(sys_cpu, 2),
                'system_memory_mb': round(sys_mem_used, 2),
                'system_memory_percent': round(sys_mem_percent, 2),
                'gpu_util_percent': round(gpu_util, 2),
                'gpu_memory_used_mb': round(gpu_mem_used, 2),
                'gpu_memory_total_mb': round(gpu_mem_total, 2),
            }
            
            with self._lock:
                self.system_data.append(record)
            
            ext_record = {
                'timestamp': timestamp,
                'elapsed_sec': round(elapsed, 3),
                'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            }
            
            for proc_name in EXTERNAL_PROCESSES:
                procs = find_processes_by_name(proc_name)
                total_cpu = 0.0
                total_mem = 0.0
                proc_count = 0
                
                for proc in procs:
                    try:
                        cpu = proc.cpu_percent(interval=None)
                        mem = proc.memory_info().rss / (1024 * 1024)
                        total_cpu += cpu
                        total_mem += mem
                        proc_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                short_name = proc_name.replace('.exe', '').lower()
                ext_record[f'{short_name}_cpu_percent'] = round(total_cpu, 2)
                ext_record[f'{short_name}_memory_mb'] = round(total_mem, 2)
                ext_record[f'{short_name}_count'] = proc_count
            
            with self._lock:
                self.external_processes_data.append(ext_record)
            
            time.sleep(self.sample_interval)
    
    def start(self):
        if self._running:
            return
        
        self.start_time = time.time()
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print(f"[PerformanceMonitor] Started monitoring PID {self.target_pid} ({self.process_name})")
        if self.gpu_name != "Unknown":
            print(f"[PerformanceMonitor] GPU: {self.gpu_name}")
    
    def stop(self):
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"[PerformanceMonitor] Stopped monitoring (duration: {duration:.1f}s)")
    
    def record_inference(self, inference_time_ms, batch_size=1, extra_info=None):
        timestamp = time.time()
        elapsed = timestamp - self.start_time if self.start_time else 0
        
        _, gpu_mem_used, _ = self._get_gpu_usage()
        
        record = {
            'timestamp': timestamp,
            'elapsed_sec': round(elapsed, 3),
            'datetime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'inference_time_ms': round(inference_time_ms, 3),
            'batch_size': batch_size,
            'throughput_per_sec': round(1000.0 / inference_time_ms if inference_time_ms > 0 else 0, 2),
            'gpu_memory_used_mb': round(gpu_mem_used, 2),
        }
        
        if extra_info:
            record.update(extra_info)
        
        with self._lock:
            self.model_data.append(record)
    
    def save_results(self, filename=None):
        os.makedirs(self.output_dir, exist_ok=True)
        
        if filename is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'performance_{timestamp_str}.xlsx'
        
        output_path = os.path.join(self.output_dir, filename)
        
        print(f"[PerformanceMonitor] Saving results...")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if self.system_data:
                df_system = pd.DataFrame(self.system_data)
                df_system.to_excel(writer, sheet_name='system_performance', index=False)
                print(f"  - System performance: {len(self.system_data)} samples")
            
            if self.model_data:
                df_model = pd.DataFrame(self.model_data)
                df_model.to_excel(writer, sheet_name='model_inference', index=False)
                print(f"  - Model inference: {len(self.model_data)} records")

            if self.external_processes_data:
                df_ext = pd.DataFrame(self.external_processes_data)
                df_ext.to_excel(writer, sheet_name='external_tools', index=False)
                print(f"  - External tools (QGC/RflySim3D/CopterSim): {len(self.external_processes_data)} samples")

            summary_data = self._generate_summary()
            df_summary = pd.DataFrame([summary_data])
            df_summary.to_excel(writer, sheet_name='summary', index=False)
        
        print(f"[PerformanceMonitor] Results saved to: {output_path}")
        return output_path
    
    def _generate_summary(self):
        summary = {
            'process_name': self.process_name,
            'process_pid': self.target_pid,
            'gpu_name': self.gpu_name,
            'monitor_start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else '',
            'monitor_duration_sec': round(time.time() - self.start_time, 2) if self.start_time else 0,
            'sample_interval_sec': self.sample_interval,
            'total_system_samples': len(self.system_data),
            'total_inference_records': len(self.model_data),
        }

        if self.system_data:
            proc_cpu = [r['process_cpu_percent'] for r in self.system_data]
            proc_mem = [r['process_memory_mb'] for r in self.system_data]
            gpu_util = [r['gpu_util_percent'] for r in self.system_data]
            gpu_mem = [r['gpu_memory_used_mb'] for r in self.system_data]
            
            summary.update({
                'avg_process_cpu_percent': round(np.mean(proc_cpu), 2),
                'max_process_cpu_percent': round(np.max(proc_cpu), 2),
                'min_process_cpu_percent': round(np.min(proc_cpu), 2),
                'avg_process_memory_mb': round(np.mean(proc_mem), 2),
                'max_process_memory_mb': round(np.max(proc_mem), 2),
                'avg_gpu_util_percent': round(np.mean(gpu_util), 2),
                'max_gpu_util_percent': round(np.max(gpu_util), 2),
                'avg_gpu_memory_mb': round(np.mean(gpu_mem), 2),
                'max_gpu_memory_mb': round(np.max(gpu_mem), 2),
            })

        if self.model_data:
            infer_times = [r['inference_time_ms'] for r in self.model_data]
            throughputs = [r['throughput_per_sec'] for r in self.model_data]
            
            summary.update({
                'total_inferences': len(self.model_data),
                'avg_inference_time_ms': round(np.mean(infer_times), 3),
                'min_inference_time_ms': round(np.min(infer_times), 3),
                'max_inference_time_ms': round(np.max(infer_times), 3),
                'std_inference_time_ms': round(np.std(infer_times), 3),
                'p50_inference_time_ms': round(np.percentile(infer_times, 50), 3),
                'p95_inference_time_ms': round(np.percentile(infer_times, 95), 3),
                'p99_inference_time_ms': round(np.percentile(infer_times, 99), 3),
                'avg_throughput_per_sec': round(np.mean(throughputs), 2),
            })

        if self.external_processes_data:
            for proc_name in EXTERNAL_PROCESSES:
                short_name = proc_name.replace('.exe', '').lower()
                cpu_key = f'{short_name}_cpu_percent'
                mem_key = f'{short_name}_memory_mb'
                
                cpu_vals = [r.get(cpu_key, 0) for r in self.external_processes_data]
                mem_vals = [r.get(mem_key, 0) for r in self.external_processes_data]
                
                if cpu_vals:
                    summary[f'avg_{short_name}_cpu_percent'] = round(np.mean(cpu_vals), 2)
                    summary[f'max_{short_name}_cpu_percent'] = round(np.max(cpu_vals), 2)
                if mem_vals:
                    summary[f'avg_{short_name}_memory_mb'] = round(np.mean(mem_vals), 2)
                    summary[f'max_{short_name}_memory_mb'] = round(np.max(mem_vals), 2)
        
        return summary
    
    def get_current_stats(self):
        if not self.system_data:
            return None
        
        with self._lock:
            latest = self.system_data[-1].copy()
        
        if self.model_data:
            with self._lock:
                recent_infer = self.model_data[-10:] if len(self.model_data) >= 10 else self.model_data
            avg_infer_time = np.mean([r['inference_time_ms'] for r in recent_infer])
            latest['recent_avg_inference_ms'] = round(avg_infer_time, 2)
        
        return latest


def monitor_external_process(process_cmd, output_dir, sample_interval=0.1):
    print(f"[PerformanceMonitor] Starting external process: {process_cmd}")

    if isinstance(process_cmd, str):
        process = subprocess.Popen(process_cmd, shell=True)
    else:
        process = subprocess.Popen(process_cmd)

    time.sleep(1.0)

    monitor = PerformanceMonitor(
        target_pid=process.pid,
        output_dir=output_dir,
        sample_interval=sample_interval
    )

    monitor.start()

    try:
        process.wait()
    except KeyboardInterrupt:
        print("[PerformanceMonitor] Interrupted by user")
        process.terminate()

    monitor.stop()
    result_path = monitor.save_results()
    
    return result_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Monitor for Python processes')
    parser.add_argument('--pid', type=int, default=None, help='Target process PID to monitor')
    parser.add_argument('--cmd', type=str, default=None, help='Command to execute and monitor')
    parser.add_argument('--duration', type=int, default=60, help='Monitoring duration in seconds (for PID mode)')
    parser.add_argument('--interval', type=float, default=0.1, help='Sampling interval in seconds')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = args.output or sys.path[0]
    
    if args.cmd:
        monitor_external_process(args.cmd, output_dir, args.interval)
    elif args.pid:
        monitor = PerformanceMonitor(
            target_pid=args.pid,
            output_dir=output_dir,
            sample_interval=args.interval
        )
        monitor.start()
        
        print(f"Monitoring PID {args.pid} for {args.duration} seconds...")
        print("Press Ctrl+C to stop early")
        try:
            for i in range(int(args.duration / args.interval)):
                time.sleep(args.interval)
                stats = monitor.get_current_stats()
                if stats and i % 10 == 0: 
                    print(f"  CPU: {stats['process_cpu_percent']:.1f}%  "
                          f"MEM: {stats['process_memory_mb']:.0f}MB  "
                          f"GPU: {stats['gpu_util_percent']:.1f}%  "
                          f"VRAM: {stats['gpu_memory_used_mb']:.0f}MB")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        monitor.stop()
        monitor.save_results()
    else:
        print("=" * 60)
        print("Performance Monitor - Demo Mode")
        print("=" * 60)
        print("Monitoring current process for 10 seconds...")
        print()
        
        monitor = PerformanceMonitor(
            output_dir=output_dir,
            sample_interval=0.1
        )
        monitor.start()

        for i in range(50):
            data = np.random.rand(500, 500)
            result = np.dot(data, data.T)

            infer_time = np.random.uniform(10, 50)  # 10-50ms
            monitor.record_inference(infer_time, batch_size=1)

            if i % 10 == 0:
                stats = monitor.get_current_stats()
                if stats:
                    print(f"[{i}/50] CPU: {stats['process_cpu_percent']:.1f}%  "
                          f"MEM: {stats['process_memory_mb']:.0f}MB  "
                          f"GPU: {stats['gpu_util_percent']:.1f}%")
            
            time.sleep(0.2)
        
        monitor.stop()
        result_path = monitor.save_results()
        
        print()
        print("=" * 60)
        print(f"Demo completed! Results saved to: {result_path}")
        print("=" * 60)
