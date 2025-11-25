#!/usr/bin/env python3
"""
RenderDoc 诊断和测试工具
用于在无 GUI 的 SSH 环境中排查 RenderDoc 捕获问题
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from typing import Tuple, Optional
import json

class RenderDocDiagnostic:
    def __init__(self):
        self.build_dir = "/home/hpf/project/vulkan/mujoco/mujoco/build"
        self.renderdoc = "/home/hpf/Downloads/renderdoc_1.41/bin/renderdoccmd"
        self.program_path = os.path.join(self.build_dir, "bin/mjbatch_smoke")
        self.output_rdc = os.path.join(self.build_dir, "capture_mjbatch.rdc")
        self.test_results = {}
        
    def print_section(self, title: str):
        """打印分节标题"""
        print(f"\n{'='*70}")
        print(f"  {title}")
        print('='*70)
    
    def run_command(self, cmd: list, description: str, 
                    capture_output: bool = True) -> Tuple[bool, str]:
        """运行命令并返回结果"""
        print(f"\n▶ {description}")
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            if capture_output:
                result = subprocess.run(cmd, capture_output=True, 
                                       text=True, timeout=30)
                success = result.returncode == 0
                output = result.stdout + result.stderr
            else:
                result = subprocess.run(cmd, timeout=30)
                success = result.returncode == 0
                output = f"返回码: {result.returncode}"
            
            status = "✅ 成功" if success else "❌ 失败"
            print(f"  {status}")
            if not success and capture_output:
                print(f"  错误输出: {result.stderr[:200]}")
            
            return success, output
        except subprocess.TimeoutExpired:
            print("  ⏱️ 超时")
            return False, "Command timeout"
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            return False, str(e)
    
    def test_1_environment_check(self):
        """测试1: 环境检查"""
        self.print_section("测试 1: 环境检查")
        
        # 检查关键文件和目录
        checks = [
            (self.renderdoc, "RenderDoc 可执行文件"),
            (self.program_path, "目标程序"),
            (self.build_dir, "构建目录"),
        ]
        
        all_exist = True
        for path, name in checks:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"{status} {name}: {path}")
            if exists and os.path.isfile(path):
                # 检查可执行权限
                is_executable = os.access(path, os.X_OK)
                exec_status = "可执行" if is_executable else "不可执行"
                print(f"   └─ {exec_status}")
            all_exist = all_exist and exists
        
        # 检查环境变量
        print("\n环境变量:")
        env_vars = ['DISPLAY', 'WAYLAND_DISPLAY', 'VK_INSTANCE_LAYERS', 
                    'VK_LAYER_PATH', 'LD_LIBRARY_PATH']
        for var in env_vars:
            value = os.environ.get(var, "未设置")
            print(f"  {var} = {value}")
        
        self.test_results['environment_check'] = all_exist
        return all_exist
    
    def test_2_renderdoc_version(self):
        """测试2: RenderDoc 版本和功能"""
        self.print_section("测试 2: RenderDoc 版本信息")
        
        success, output = self.run_command(
            [self.renderdoc, "--version"],
            "获取 RenderDoc 版本"
        )
        
        if success:
            print(f"\n版本信息:\n{output}")
        
        # 测试 help 命令
        success2, output2 = self.run_command(
            [self.renderdoc, "--help"],
            "获取 RenderDoc 帮助信息"
        )
        
        if success2:
            print(f"\n可用命令:\n{output2[:500]}")
        
        self.test_results['renderdoc_version'] = success
        return success
    
    def test_3_vulkan_support(self):
        """测试3: Vulkan 支持检查"""
        self.print_section("测试 3: Vulkan 支持")
        
        # 检查 vulkaninfo
        vulkaninfo_path = shutil.which("vulkaninfo")
        if vulkaninfo_path:
            success, output = self.run_command(
                ["vulkaninfo", "--summary"],
                "Vulkan 设备信息"
            )
            if success:
                print(f"\nVulkan 信息:\n{output[:500]}")
        else:
            print("❌ 未找到 vulkaninfo 工具")
            success = False
        
        # 检查 Vulkan layer
        layer_path = "/usr/share/vulkan/explicit_layer.d"
        if os.path.exists(layer_path):
            layers = os.listdir(layer_path)
            print(f"\n可用 Vulkan layers ({layer_path}):")
            for layer in layers:
                print(f"  - {layer}")
        
        self.test_results['vulkan_support'] = success
        return success
    
    def test_4_simple_capture(self):
        """测试4: 简单的捕获测试（使用 vkcube 等简单程序）"""
        self.print_section("测试 4: 简单程序捕获测试")
        
        # 尝试找到简单的测试程序
        test_programs = [
            ("vkcube", ["vkcube", "--c", "1"]),  # 运行1帧后退出
            ("vulkaninfo", ["vulkaninfo"]),
        ]
        
        for prog_name, cmd in test_programs:
            prog_path = shutil.which(prog_name)
            if not prog_path:
                print(f"⊘ 跳过 {prog_name} (未安装)")
                continue
            
            test_output = os.path.join(self.build_dir, f"test_{prog_name}.rdc")
            
            # 尝试捕获
            capture_cmd = [
                self.renderdoc, "capture",
                "--working-dir", self.build_dir,
                "--output", test_output,
                "--", prog_path
            ] + cmd[1:] if len(cmd) > 1 else [prog_path]
            
            success, output = self.run_command(
                capture_cmd,
                f"捕获 {prog_name}",
                capture_output=True
            )
            
            if success and os.path.exists(test_output):
                size = os.path.getsize(test_output)
                print(f"  生成文件: {test_output} ({size} bytes)")
                self.test_results[f'simple_capture_{prog_name}'] = True
                return True
            else:
                print(f"  输出: {output[:300]}")
        
        self.test_results['simple_capture'] = False
        return False
    
    def test_5_headless_mode(self):
        """测试5: 不同的无头模式配置"""
        self.print_section("测试 5: 无头模式配置测试")
        
        # 测试不同的环境变量组合
        test_configs = [
            {
                "name": "默认配置",
                "env": {}
            },
            {
                "name": "XVFB 虚拟显示",
                "env": {"DISPLAY": ":99"}
            },
            {
                "name": "强制软件渲染",
                "env": {
                    "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
                    "LIBGL_ALWAYS_SOFTWARE": "1"
                }
            },
            {
                "name": "RenderDoc 全局钩子",
                "env": {"RENDERDOC_HOOK_EGL": "1"}
            }
        ]
        
        for config in test_configs:
            print(f"\n--- 配置: {config['name']} ---")
            test_output = os.path.join(
                self.build_dir, 
                f"test_{config['name'].replace(' ', '_')}.rdc"
            )
            
            env = os.environ.copy()
            env.update(config['env'])
            
            try:
                result = subprocess.run(
                    [
                        self.renderdoc, "capture",
                        "--working-dir", self.build_dir,
                        "--output", test_output,
                        "--", self.program_path
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                success = result.returncode == 0 and os.path.exists(test_output)
                status = "✅" if success else "❌"
                print(f"{status} {config['name']}")
                
                if not success:
                    print(f"  stderr: {result.stderr[:200]}")
                elif os.path.exists(test_output):
                    size = os.path.getsize(test_output)
                    print(f"  生成文件: {size} bytes")
                    
                self.test_results[f'headless_{config["name"]}'] = success
                
            except Exception as e:
                print(f"❌ 异常: {e}")
                self.test_results[f'headless_{config["name"]}'] = False
        
        return any(v for k, v in self.test_results.items() if k.startswith('headless_'))
    
    def test_6_program_direct_run(self):
        """测试6: 直接运行目标程序（不使用 RenderDoc）"""
        self.print_section("测试 6: 直接运行目标程序")
        
        success, output = self.run_command(
            [self.program_path],
            "直接运行 mjbatch_smoke",
            capture_output=True
        )
        
        if success:
            print(f"\n程序输出:\n{output[:500]}")
            # 检查是否生成了预期的输出文件
            output_file = os.path.join(self.build_dir, "humanoid.ppm")
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"\n✅ 生成输出文件: {output_file} ({size} bytes)")
        
        self.test_results['program_direct_run'] = success
        return success
    
    def test_7_alternative_capture_methods(self):
        """测试7: 替代的捕获方法"""
        self.print_section("测试 7: 替代捕获方法")
        
        # 方法1: 使用 remoteserver
        print("\n方法1: RenderDoc Remote Server")
        print("  说明: 在无 GUI 环境中，可以启动 remoteserver")
        print("  命令: renderdoccmd remoteserver")
        print("  然后从本地 RenderDoc GUI 连接进行捕获")
        
        # 方法2: 使用 VK_LAYER
        print("\n方法2: Vulkan Layer 捕获")
        print("  设置环境变量:")
        print("    export VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture")
        print("    export RENDERDOC_CAPTUREOPTS=AllowVSync=1")
        
        # 方法3: 检查是否可以使用 apitrace
        apitrace = shutil.which("apitrace")
        if apitrace:
            print("\n方法3: 使用 apitrace (替代工具)")
            print(f"  已安装 apitrace: {apitrace}")
        
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("╔═══════════════════════════════════════════════════════════════╗")
        print("║         RenderDoc 诊断工具 - 无 GUI 环境测试                 ║")
        print("╚═══════════════════════════════════════════════════════════════╝")
        
        tests = [
            self.test_1_environment_check,
            self.test_2_renderdoc_version,
            self.test_3_vulkan_support,
            self.test_6_program_direct_run,
            self.test_4_simple_capture,
            self.test_5_headless_mode,
            self.test_7_alternative_capture_methods,
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"\n❌ 测试异常: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成测试报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        self.print_section("测试报告总结")
        
        print("\n测试结果:")
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {status} - {test_name}")
        
        passed = sum(1 for v in self.test_results.values() if v)
        total = len(self.test_results)
        
        print(f"\n总计: {passed}/{total} 测试通过")
        
        # 生成建议
        print("\n" + "="*70)
        print("建议和解决方案:")
        print("="*70)
        
        if not self.test_results.get('environment_check'):
            print("\n⚠️  环境问题:")
            print("  - 检查所有文件路径是否正确")
            print("  - 确保 RenderDoc 和目标程序有执行权限")
        
        if not self.test_results.get('vulkan_support'):
            print("\n⚠️  Vulkan 支持问题:")
            print("  - 安装 vulkan-tools: sudo apt install vulkan-tools")
            print("  - 检查显卡驱动是否支持 Vulkan")
        
        if not any(v for k, v in self.test_results.items() if 'capture' in k):
            print("\n⚠️  捕获失败的可能原因:")
            print("  1. 无 GPU 环境: 考虑使用软件渲染 (lavapipe)")
            print("     sudo apt install mesa-vulkan-drivers")
            print("  2. SSH 无显示: 使用 Xvfb 虚拟显示")
            print("     sudo apt install xvfb")
            print("     Xvfb :99 -screen 0 1024x768x24 &")
            print("     export DISPLAY=:99")
            print("  3. 使用 RenderDoc Remote Server 模式")
            print("     在服务器: renderdoccmd remoteserver")
            print("     在本地: 通过 GUI 连接远程服务器")
        
        # 保存报告到文件
        report_file = os.path.join(self.build_dir, "renderdoc_diagnostic_report.json")
        try:
            with open(report_file, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n📄 详细报告已保存到: {report_file}")
        except Exception as e:
            print(f"\n⚠️  无法保存报告: {e}")

def main():
    diagnostic = RenderDocDiagnostic()
    diagnostic.run_all_tests()

if __name__ == "__main__":
    main()