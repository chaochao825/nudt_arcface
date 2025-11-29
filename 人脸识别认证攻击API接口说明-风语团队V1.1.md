
## 1. 生成对抗样本节点 API (SSE流式响应)

### 1.1 生成人脸识别对抗样本

**URL**: `127.0.0.1:19001/api-ai-server/face-recognition-attack/generate-adversarial-v1`
**Method**: POST
**INPUT**:

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306",
    "method_type": "人脸识别攻击",
    "algorithm_type": "BIM攻击",
    "task_type": "样本生成",
    "task_name": "人脸识别BIM对抗样本生成",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "generation_config": {
      "attack_algorithm": "BIM",
      "dataset_config": {
        "dataset_name": "LFW",
        "dataset_format": "image",
        "total_samples": 13233,
        "selected_samples": 150,
        "sample_selection_strategy": "random"
      },
      "algorithm_parameters": {
        "epsilon": 0.08,
        "step_size": 0.002,
        "max_iterations": 50,
        "targeted": false,
        "target_class": -1,
        "random_start": false,
        "loss_function": "cosine_similarity",
        "optimization_method": "gradient_ascent",
        "momentum": 0.90
      },
      "constraints": {
        "perturbation_norm": "linf",
        "max_perturbation": 0.08,
        "clip_min": 0.0,
        "clip_max": 1.0,
        "spatial_constraints": {
          "enabled": false,
          "mask_regions": []
        }
      }
    },
    "model_config": {
      "model_name": "DeepFace",
      "model_state": "loaded"
    },
    "monitoring_config": {
      "real_time_metrics": [
        "generation_progress",
        "current_perturbation_norm",
        "success_rate_current",
        "memory_usage",
        "computation_time"
      ],
      "quality_metrics": [
        "visual_quality",
        "perturbation_visibility",
        "attack_effectiveness"
      ]
    }
  }
}
```
# 人脸识别对抗样本生成API接口入参说明

## 回调参数 (callback_params)

- **task_run_id**: 任务运行唯一标识符，采用UUID格式，用于追踪和管理任务执行流程
- **method_type**: 方法类型分类，标识当前任务为"人脸识别攻击"类别
- **algorithm_type**: 具体算法类型，指定使用"BIM攻击"算法
- **task_type**: 任务类型分类，定义任务为"样本生成"操作类型
- **task_name**: 具体任务名称，详细描述为"人脸识别BIM对抗样本生成"
- **parent_task_id**: 父任务标识符，用于任务链的关联和管理，关联上级任务
- **user_name**: 执行用户名称，记录任务执行者信息

## 业务参数 (business_params)

- **user_name**: 执行用户名称，记录任务执行者信息
- **scene_instance_id**: 场景实例标识符，用于标识当前攻击场景实例
- **generation_config**: 对抗样本生成配置
  - **attack_algorithm**: 攻击算法名称，指定使用BIM算法
  - **dataset_config**: 数据集配置
    - **dataset_name**: 数据集名称，使用LFW人脸数据集
    - **dataset_format**: 数据集格式，指定为图像格式
    - **total_samples**: 数据集总样本数
    - **selected_samples**: 选择用于攻击的样本数量
    - **sample_selection_strategy**: 样本选择策略，采用随机选择方式
  - **algorithm_parameters**: 算法参数配置
    - **epsilon**: 扰动大小限制参数，控制对抗扰动的最大幅度
    - **step_size**: 迭代步长参数，控制每次迭代的扰动幅度
    - **max_iterations**: 最大迭代次数，限制算法运行轮数
    - **targeted**: 攻击类型标识，false表示非目标攻击
    - **target_class**: 目标类别，-1表示非目标攻击
    - **random_start**: 随机起始点标识，控制是否从随机扰动开始
    - **loss_function**: 损失函数类型，使用余弦相似度作为优化目标
    - **optimization_method**: 优化方法，采用梯度上升策略
    - **momentum**: 动量因子，加速收敛并减少震荡
  - **constraints**: 扰动约束条件
    - **perturbation_norm**: 扰动范数类型，使用L∞范数约束
    - **max_perturbation**: 最大扰动限制，确保扰动不可见
    - **clip_min**: 像素值裁剪下限，保持图像有效性
    - **clip_max**: 像素值裁剪上限，保持图像有效性
    - **spatial_constraints**: 空间约束配置
      - **enabled**: 空间约束启用标识
      - **mask_regions**: 掩码区域列表，指定保护区域
- **model_config**: 目标模型配置
  - **model_name**: 目标模型名称，使用DeepFace人脸识别模型
  - **model_state**: 模型状态标识，表示模型已加载就绪
- **monitoring_config**: 监控配置
  - **real_time_metrics**: 实时监控指标列表，包括生成进度、当前扰动范数、当前成功率、内存使用、计算时间等
  - **quality_metrics**: 质量评估指标列表，包括视觉质量、扰动可见性、攻击有效性等


**OUTPUT** (SSE流式响应):

```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:00:123", "data": {"event": "process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始人脸识别BIM对抗样本生成任务", "log": "[5%] 开始人脸识别BIM对抗样本生成任务\n", "details": {"attack_method": "BIM", "target_model": "DeepFace", "max_samples": 150, "business_params": {"scene_instance_id": "f54d72a78c264f9bb936954522881e7c", "model_info": {"model_name": "DeepFace", "model_path": "/models/deepface/weights.h5"}, "dataset_info": {"dataset_name": "LFW", "dataset_path": "/datasets/lfw/processed", "sample_type": "face_image", "total_samples": 13233, "sample_indices": "random_150"}, "generation_config": {"method_type": "人脸识别攻击", "algorithm_type": "BIM", "adversarial_samples_dir": "/data/adversarial_samples/face_recognition_123456789", "adversarial_samples_name": "adversarial_samples_face_bim", "original_samples_save_path": "/data/original_samples/face_recognition_123456789", "original_samples_name": "original_samples_face", "visualization_dir": "/data/visualizations/face_recognition_123456789", "visualization_name": "perturbation_visualization_face", "file_format": "npy", "max_samples": 150, "save_visualizations": true, "sample_selection_strategy": "random", "user_params": {"epsilon": 0.08, "targeted": false, "target_class": null, "norm_type": "inf", "max_iterations": 50, "step_size": 0.002, "momentum": 0.90}}}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:01:456", "data": {"event": "model_loaded", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "人脸识别模型加载成功", "log": "[15%] 目标模型DeepFace加载完成\n", "details": {"model_name": "DeepFace", "model_path": "/models/deepface/weights.h5", "model_type": "face_recognition", "input_shape": [160, 160, 3], "embedding_dim": 128, "recognition_threshold": 0.6}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:02:789", "data": {"event": "dataset_loaded", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "人脸数据集加载完成", "log": "[25%] 加载150个人脸图像样本\n", "details": {"dataset_path": "/datasets/lfw/processed", "sample_count": 150, "dataset_type": "face_image", "sample_indices": "random_150", "dataset_info": {"name": "LFW", "total_samples": 13233, "identity_count": 5749, "avg_samples_per_identity": 2.3}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:03:123", "data": {"event": "generation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 30, "message": "开始生成人脸对抗样本", "log": "[30%] 开始BIM对抗样本生成，扰动参数epsilon=0.08, 迭代次数=50\n", "details": {"attack_params": {"epsilon": 0.08, "targeted": false, "norm_type": "inf", "max_iterations": 50, "step_size": 0.002, "momentum": 0.90}, "batch_size": 20}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:15:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 40, "message": "人脸样本处理中", "log": "[40%] 正在处理样本30/150 - 计算特征梯度\n", "details": {"current_sample": 30, "total_samples": 150, "step": "feature_gradient_computation", "batch_progress": "1/8", "current_success_rate": 0.72}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:27:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 50, "message": "人脸样本处理中", "log": "[50%] 正在处理样本60/150 - 应用人脸扰动\n", "details": {"current_sample": 60, "total_samples": 150, "step": "face_perturbation_application", "batch_progress": "2/8", "current_success_rate": 0.75}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:40:123", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 60, "message": "人脸样本处理中", "log": "[60%] 正在处理样本90/150 - 验证人脸识别攻击效果\n", "details": {"current_sample": 90, "total_samples": 150, "step": "face_recognition_validation", "batch_progress": "3/8", "current_success_rate": 0.78}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:15:52:456", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 70, "message": "人脸样本处理中", "log": "[70%] 正在处理样本120/150 - 优化人脸特征扰动\n", "details": {"current_sample": 120, "total_samples": 150, "step": "face_feature_optimization", "batch_progress": "4/8", "current_success_rate": 0.80}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:04:789", "data": {"event": "sample_processing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 80, "message": "人脸样本处理中", "log": "[80%] 正在处理样本150/150 - 最终人脸验证\n", "details": {"current_sample": 150, "total_samples": 150, "step": "final_face_validation", "batch_progress": "5/8", "current_success_rate": 0.82}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:05:123", "data": {"event": "generation_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 85, "message": "人脸对抗样本生成完成", "log": "[85%] 人脸对抗样本生成完成 - 成功生成123/150样本\n", "details": {"total_samples": 150, "successful_samples": 123, "success_rate": 0.82, "generation_time": "65.0秒", "avg_perturbation": 0.052, "attack_success_rate": 0.82}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:10:456", "data": {"event": "visualization_generated", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "人脸扰动可视化生成完成", "log": "[90%] 人脸扰动可视化图像已生成\n", "details": {"visualization_path": "/data/visualizations/face_recognition_123456789/perturbation_visualization_face", "file_format": "png", "sample_count": 123, "visualization_types": ["original_vs_adversarial", "face_perturbation_heatmap", "recognition_comparison"]}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:15:789", "data": {"event": "results_saved", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 95, "message": "人脸对抗样本结果保存完成", "log": "[95%] 人脸对抗样本结果保存完成\n", "details": {"output_files": {"adversarial_samples": "/data/adversarial_samples/face_recognition_123456789/adversarial_samples_face_bim.npy", "original_samples": "/data/original_samples/face_recognition_123456789/original_samples_face.npy", "visualization_files": "/data/visualizations/face_recognition_123456789/perturbation_visualization_face.zip", "metadata_file": "/data/adversarial_samples/face_recognition_123456789/generation_metadata.json"}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-10:16:20:123", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3306", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "样本生成", "task_name": "人脸识别BIM对抗样本生成", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "人脸识别BIM对抗样本生成任务完成", "log": "[100%] 人脸识别BIM对抗样本生成任务完成\n", "details": {"generation_id": "gen_bim_face_202407011015", "attack_method": "BIM", "attack_type": "白盒", "generation_stats": {"total_samples": 150, "successful_samples": 123, "success_rate": 0.82, "attack_success_rate": 0.82, "avg_perturbation_magnitude": 0.052, "generation_time": "80.0秒"}, "quality_metrics": {"avg_l2_norm": 3.25, "avg_linf_norm": 0.052, "original_recognition_rate": 0.98, "adversarial_recognition_rate": 0.18, "psnr": 35.8, "ssim": 0.94}, "output_files": {"adversarial_samples": "adversarial_samples_face_bim.zip", "original_samples": "original_samples_face.zip", "visualization_files": "perturbation_visualization_face.zip", "metadata_file": "generation_metadata.json"}, "adversarial_samples_info": {"sample_count": 123, "format": "numpy_array", "dimensions": [123, 160, 160, 3], "data_type": "float32", "perturbation_range": [0.04, 0.08]}, "original_dataset": "LFW"}}}
```

# SSE消息details参数简要说明

## 基本信息
- **generation_id**: 生成任务唯一ID，包含时间信息
- **attack_method**: 使用的攻击算法（BIM）
- **attack_type**: 攻击类型（白盒攻击）

## 生成统计
- **total_samples**: 总处理样本数（150个）
- **successful_samples**: 成功生成样本数（123个）
- **success_rate**: 生成成功率（82%）
- **attack_success_rate**: 攻击成功率（82%）
- **avg_perturbation_magnitude**: 平均扰动幅度（0.052）
- **generation_time**: 生成耗时（80秒）

## 质量指标
- **avg_l2_norm**: 平均L2范数（3.25）
- **avg_linf_norm**: 平均L∞范数（0.052）
- **original_recognition_rate**: 原始识别率（98%）
- **adversarial_recognition_rate**: 对抗样本识别率（18%）
- **psnr**: 峰值信噪比（35.8dB，质量良好）
- **ssim**: 结构相似性（0.94，结构保持好）

## 输出文件
- **adversarial_samples**: 对抗样本文件
- **original_samples**: 原始样本文件  
- **visualization_files**: 可视化文件
- **metadata_file**: 元数据文件

## 样本信息
- **sample_count**: 样本数量（123个）
- **format**: 数据格式（numpy数组）
- **dimensions**: 数据维度[123,160,160,3]
- **data_type**: 数据类型（float32）
- **perturbation_range**: 扰动范围[0.04,0.08]

## 原始数据
- **original_dataset**: 使用数据集（LFW人脸数据集）


## 2. 执行攻击节点 API (SSE流式响应)

### 2.1 执行人脸识别攻击

**URL**: `127.0.0.1:19001/api-ai-server/face-recognition-attack/execute-face-attack-v1`
**Method**: POST
**INPUT**:

```json
{
  "callback_params": {
    "task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307",
    "method_type": "人脸识别攻击",
    "algorithm_type": "BIM攻击",
    "task_type": "攻击执行",
    "task_name": "人脸识别BIM攻击",
    "parent_task_id": "f54d72a78c264f9bb93695f522881e7c",
    "user_name": "zhangxueyou"
  },
  "business_params": {
    "user_name": "zhangxueyou",
    "scene_instance_id": "f54d72a78c264f9bb936954522881e7c",
    "attack_config": {
      "attack_method": "BIM",
      "adversarial_samples_name": "adversarial_samples_face_bim.zip",
      "original_samples_name": "original_samples_face.zip",
      "evaluation_mode": "comparison",
      "comparison_baseline": "original"
    },
    "evaluation_config": {
      "attack_effectiveness_metrics": {
        "recognition_metrics": {
          "original_recognition": true,
          "adversarial_recognition": true,
          "recognition_drop_rate": true,
          "confidence_reduction": true,
          "false_acceptance_rate": true,
          "false_rejection_rate": true
        },
        "verification_metrics": {
          "verification_accuracy_change": true,
          "equal_error_rate_change": true,
          "roc_curve_analysis": true
        },
        "identification_metrics": {
          "rank1_accuracy_change": true,
          "identification_failure_rate": true
        }
      },
      "perturbation_metrics": {
        "norm_metrics": {
          "l0_norm": true,
          "l2_norm": true,
          "linf_norm": true,
          "psnr": true,
          "ssim": true
        },
        "visual_metrics": {
          "human_perceptibility": true,
          "face_quality_assessment": true,
          "structural_similarity": true
        }
      },
      "performance_metrics": {
        "inference_time": true,
        "throughput": true,
        "memory_usage": true,
        "computational_cost": true
      }
    },
    "monitoring_config": {
      "real_time_metrics": [
        "current_success_rate",
        "recognition_accuracy_drop",
        "confidence_reduction",
        "perturbation_visibility",
        "processing_throughput"
      ]
    }
  }
}
```

# 人脸识别攻击执行API接口入参说明

## 回调参数 (callback_params)

- **task_run_id**: 任务运行唯一标识符，采用UUID格式，用于追踪和管理任务执行流程
- **method_type**: 方法类型分类，标识当前任务为"人脸识别攻击"类别
- **algorithm_type**: 具体算法类型，指定使用"BIM攻击"算法
- **task_type**: 任务类型分类，定义任务为"攻击执行"操作类型
- **task_name**: 具体任务名称，详细描述为"人脸识别BIM攻击"
- **parent_task_id**: 父任务标识符，用于任务链的关联和管理，关联上级任务
- **user_name**: 执行用户名称，记录任务执行者信息

## 业务参数 (business_params)

- **user_name**: 执行用户名称，记录任务执行者信息
- **scene_instance_id**: 场景实例标识符，用于标识当前攻击场景实例
- **attack_config**: 攻击执行配置
  - **attack_method**: 攻击方法名称，指定使用BIM算法
  - **adversarial_samples_name**: 对抗样本文件名称，指定已生成的对抗样本文件
  - **original_samples_name**: 原始样本文件名称，指定原始样本文件
  - **evaluation_mode**: 评估模式，设置为对比评估模式
  - **comparison_baseline**: 对比基线，以原始样本作为对比基准
- **evaluation_config**: 攻击效果评估配置
  - **attack_effectiveness_metrics**: 攻击有效性评估指标
    - **recognition_metrics**: 识别相关指标
      - **original_recognition**: 原始识别率，评估原始样本的识别性能
      - **adversarial_recognition**: 对抗样本识别率，评估攻击后的识别性能
      - **recognition_drop_rate**: 识别率下降幅度，计算识别性能下降比例
      - **confidence_reduction**: 置信度降低程度，评估模型置信度变化
      - **false_acceptance_rate**: 误接受率，评估安全性能下降
      - **false_rejection_rate**: 误拒绝率，评估可用性能下降
    - **verification_metrics**: 验证相关指标
      - **verification_accuracy_change**: 验证准确率变化，评估1:1验证性能
      - **equal_error_rate_change**: 等错误率变化，评估验证系统阈值性能
      - **roc_curve_analysis**: ROC曲线分析，全面评估验证系统性能
    - **identification_metrics**: 辨识相关指标
      - **rank1_accuracy_change**: Rank-1准确率变化，评估1:N辨识性能
      - **identification_failure_rate**: 辨识失败率，评估系统可靠性
  - **perturbation_metrics**: 扰动分析指标
    - **norm_metrics**: 范数指标
      - **l0_norm**: L0范数，评估扰动的稀疏性
      - **l2_norm**: L2范数，评估扰动的总体幅度
      - **linf_norm**: L∞范数，评估扰动的最大变化
      - **psnr**: 峰值信噪比，评估图像质量保持程度
      - **ssim**: 结构相似性指数，评估结构信息保持程度
    - **visual_metrics**: 视觉质量指标
      - **human_perceptibility**: 人类可感知性，评估扰动的视觉隐蔽性
      - **face_quality_assessment**: 人脸质量评估，评估人脸图像质量变化
      - **structural_similarity**: 结构相似性，评估图像结构完整性
  - **performance_metrics**: 性能指标
    - **inference_time**: 推理时间，评估攻击对推理速度的影响
    - **throughput**: 吞吐量，评估系统处理能力
    - **memory_usage**: 内存使用量，评估资源消耗
    - **computational_cost**: 计算成本，评估整体计算开销
- **monitoring_config**: 实时监控配置
  - **real_time_metrics**: 实时监控指标列表，包括当前成功率、识别准确率下降、置信度降低、扰动可见性、处理吞吐量等


**OUTPUT** (SSE流式响应):

```json
data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:00:123", "data": {"event": "attack_process_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 5, "message": "开始人脸识别BIM攻击执行任务", "log": "[5%] 开始人脸识别BIM攻击执行任务 - 目标模型: DeepFace\n", "details": {"attack_method": "BIM", "target_model": "DeepFace", "total_samples": 123, "batch_size": 10, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:01:456", "data": {"event": "sample_loading", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 15, "message": "人脸攻击样本加载中", "log": "[15%] 加载原始人脸样本: original_samples_face.zip，对抗样本: adversarial_samples_face_bim.zip\n", "details": {"original_samples": "original_samples_face.zip", "adversarial_samples": "adversarial_samples_face_bim.zip", "total_available_samples": 123, "selected_sample_count": 123, "sample_categories": ["face_recognition"], "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:02:789", "data": {"event": "sample_preprocessing", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 25, "message": "人脸样本预处理完成", "log": "[25%] 人脸样本预处理完成 - 人脸检测、对齐、标准化\n", "details": {"preprocessing_steps": ["face_detection", "face_alignment", "normalization_160x160"], "final_sample_count": 123, "identity_count": 85, "samples_per_identity": {"min": 1, "max": 3, "avg": 1.45}, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:03:123", "data": {"event": "model_loading", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 35, "message": "人脸识别模型加载完成", "log": "[35%] DeepFace模型加载成功，准备执行人脸识别\n", "details": {"target_model": "DeepFace", "model_version": "v1.0", "input_size": "160x160", "recognition_threshold": 0.6, "embedding_dim": 128, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:05:456", "data": {"event": "attack_initialization", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 45, "message": "BIM攻击初始化完成", "log": "[45%] BIM攻击参数配置完成 - epsilon: 0.08, 迭代次数: 50\n", "details": {"attack_technique": "Basic Iterative Method", "epsilon": 0.08, "iterations": 50, "target_model_state": "loaded", "memory_usage": "1.8GB", "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:06:789", "data": {"event": "baseline_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 50, "message": "开始基线人脸识别评估", "log": "[50%] 对原始人脸样本进行基线识别评估\n", "details": {"evaluation_phase": "baseline", "samples_processed": 0, "total_samples": 123, "current_batch": 1, "total_batches": 13, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:08:123", "data": {"event": "baseline_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 55, "message": "基线评估进行中", "log": "[55%] 批次1/13 - 原始人脸样本识别完成，平均置信度: 0.92\n", "details": {"current_batch": 1, "total_batches": 13, "batch_size": 10, "average_confidence": 0.92, "recognition_count": 10, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:12:456", "data": {"event": "adversarial_evaluation_start", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 60, "message": "开始对抗样本人脸识别评估", "log": "[60%] 基线评估完成，开始对抗样本人脸识别评估\n", "details": {"evaluation_phase": "adversarial", "baseline_results": {"total_recognition": 121, "recognition_rate": 0.98, "average_confidence": 0.91}, "current_batch": 1, "total_batches": 13, "attack_statistics": {"successful_attacks": 0, "defended_samples": 0, "average_attack_time": 0.0, "current_attack_intensity": 0}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:15:789", "data": {"event": "adversarial_evaluation_progress", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 65, "message": "对抗样本评估进行中", "log": "[65%] 批次3/13 - 对抗样本识别，识别率显著下降\n", "details": {"current_batch": 3, "total_batches": 13, "recognition_drop_rate": 0.68, "confidence_reduction": 0.45, "successful_attacks": 22, "attack_statistics": {"successful_attacks": 22, "defended_samples": 8, "average_attack_time": 0.18, "current_attack_intensity": 73}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:18:123", "data": {"event": "real_time_metrics", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 70, "message": "实时监控指标更新", "log": "[70%] 实时指标 - 攻击成功率: 68%, 识别率下降: 68%, 扰动可见性: 低\n", "details": {"current_success_rate": 0.68, "recognition_accuracy_drop": 0.68, "confidence_reduction": 0.45, "perturbation_visibility": "low", "processing_throughput": "18.5 samples/sec", "attack_statistics": {"successful_attacks": 47, "defended_samples": 23, "average_attack_time": 0.19, "current_attack_intensity": 67}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:22:456", "data": {"event": "perturbation_analysis", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 75, "message": "人脸扰动分析完成", "log": "[75%] 人脸扰动指标计算完成 - L2范数: 0.018, PSNR: 35.8dB\n", "details": {"l2_norm": 0.018, "linf_norm": 0.08, "psnr": 35.8, "ssim": 0.94, "human_perceptibility": "imperceptible", "face_quality_score": 0.89, "attack_statistics": {"successful_attacks": 62, "defended_samples": 28, "average_attack_time": 0.20, "current_attack_intensity": 69}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:25:789", "data": {"event": "mid_process_summary", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 80, "message": "攻击评估中期汇总", "log": "[80%] 已完成85个样本评估，当前攻击成功率: 68%\n", "details": {"processed_samples": 85, "total_samples": 123, "current_success_rate": 0.68, "identity_breakdown": {"identity_1": 0.75, "identity_2": 0.67, "identity_3": 0.62, "identity_4": 0.70}, "attack_statistics": {"successful_attacks": 58, "defended_samples": 27, "average_attack_time": 0.19, "current_attack_intensity": 68}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:28:123", "data": {"event": "final_evaluation_batch", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 90, "message": "最终批次评估", "log": "[90%] 处理最终批次13/13，完成所有样本评估\n", "details": {"current_batch": 13, "total_batches": 13, "remaining_samples": 8, "estimated_completion_time": "1分钟", "attack_statistics": {"successful_attacks": 82, "defended_samples": 31, "average_attack_time": 0.21, "current_attack_intensity": 73}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:30:456", "data": {"event": "attack_execution_completed", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 95, "message": "人脸识别攻击执行完成", "log": "[95%] BIM攻击评估完成 - 所有123个样本处理完毕\n", "details": {"total_samples": 123, "successful_attacks": 90, "failed_attacks": 33, "overall_success_rate": 0.73, "total_execution_time": "25.3秒", "attack_statistics": {"successful_attacks": 90, "defended_samples": 33, "average_attack_time": 0.21, "current_attack_intensity": 73}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:32:789", "data": {"event": "results_analysis", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 98, "message": "人脸识别攻击结果分析完成", "log": "[98%] 人脸识别攻击结果分析完成，生成详细评估报告\n", "details": {"attack_effectiveness": 0.73, "model_vulnerability_score": 0.75, "recognition_metrics": {"recognition_drop_rate": 0.68, "confidence_reduction": 0.45, "false_acceptance_rate": 0.08, "false_rejection_rate": 0.25}, "perturbation_metrics": {"l2_norm": 0.018, "psnr": 35.8, "ssim": 0.94, "face_quality_score": 0.89}, "attack_statistics": {"successful_attacks": 90, "defended_samples": 33, "average_attack_time": 0.21, "current_attack_intensity": 73}}}}

data: {"resp_code": 0, "resp_msg": "操作成功", "time_stamp": "2024/07/01-14:35:34:123", "data": {"event": "final_result", "callback_params": {"task_run_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3307", "method_type": "人脸识别攻击", "algorithm_type": "BIM攻击", "task_type": "攻击执行", "task_name": "人脸识别BIM攻击", "parent_task_id": "f54d72a78c264f9bb93695f522881e7c", "user_name": "zhangxueyou"}, "progress": 100, "message": "人脸识别BIM攻击任务完成", "log": "[100%] 人脸识别BIM攻击任务完成\n", "details": {"execution_id": "face_attack_202407011435", "attack_method": "BIM", "target_model": "DeepFace", "execution_stats": {"total_samples": 123, "successful_attacks": 90, "success_rate": 0.73, "average_inference_time": "0.21秒", "total_execution_time": "29.0秒"}, "effectiveness_analysis": {"recognition_drop_rate": 0.68, "confidence_reduction": 0.45, "false_acceptance_rate": 0.08, "false_rejection_rate": 0.25, "equal_error_rate_change": 0.15}, "perturbation_analysis": {"l2_norm": 0.018, "linf_norm": 0.08, "psnr": 35.8, "ssim": 0.94, "human_perceptibility": "imperceptible", "face_quality_score": 0.89}, "attack_statistics": {"successful_attacks": 90, "defended_samples": 33, "average_attack_time": 0.21, "current_attack_intensity": 73}}}}
```

# SSE消息details参数简要说明

## 基本信息
- **execution_id**: 攻击执行任务ID，包含时间信息
- **attack_method**: 使用的攻击算法（BIM）
- **target_model**: 攻击目标模型（DeepFace）

## 执行统计
- **total_samples**: 总测试样本数（123个）
- **successful_attacks**: 成功攻击数（90个）
- **success_rate**: 攻击成功率（73%）
- **average_inference_time**: 平均推理时间（0.21秒）
- **total_execution_time**: 总执行时间（29秒）

## 攻击效果分析
- **recognition_drop_rate**: 识别率下降幅度（68%）
- **confidence_reduction**: 置信度降低程度（45%）
- **false_acceptance_rate**: 误接受率（8%）
- **false_rejection_rate**: 误拒绝率（25%）
- **equal_error_rate_change**: 等错误率变化（15%）

## 扰动分析
- **l2_norm**: L2范数（0.018）
- **linf_norm**: L∞范数（0.08）
- **psnr**: 峰值信噪比（35.8dB）
- **ssim**: 结构相似性（0.94）
- **human_perceptibility**: 人类可感知性（不可察觉）
- **face_quality_score**: 人脸质量评分（0.89）

## 攻击统计
- **successful_attacks**: 成功攻击次数（90次）
- **defended_samples**: 防御成功样本数（33个）
- **average_attack_time**: 平均攻击时间（0.21秒）
- **current_attack_intensity**: 当前攻击强度（73%）