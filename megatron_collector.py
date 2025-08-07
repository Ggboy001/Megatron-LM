#############################################
### Megatron-Core Collector for vtimeline ###
#############################################

import os
import json
import torch
import duckdb
import hashlib


def _get_cksum(data: torch.Tensor):
    byte_data = data.detach().cpu().view(torch.uint8).contiguous().numpy().tobytes()
    hasher = hashlib.sha256()
    hasher.update(byte_data)

    return hasher.hexdigest()


class MegatronCollector:
    step_ = 0
    dump_step_: int = int(os.getenv("VTIMELINE_DUMP_STEP", -1))

    def __init__(self):
        raise RuntimeError("Use initialize to init Megatron Core Collector")

    @classmethod
    def initialize(cls):
        root_dir = os.environ.get("VTIMELINE_LOGGER_DIR", "$HOME/sly/code/Megatron-LM/db")
        db_dir = os.path.join(root_dir, "Collector")
        os.makedirs(db_dir, exist_ok=True)

        assert hasattr(cls, "ranks_info_"), "the rank information must be set"

        db_path = os.path.join(
            root_dir,
            "Collector/coredump_{}_{}_{}_{}.db".format(
                cls.ranks_info_["dp"], cls.ranks_info_["tp"],cls.ranks_info_["pp"],cls.ranks_info_["cp"],
            ),
        )
        cls.db_ = duckdb.connect(db_path)

        cls.db_.execute(
            """CREATE TABLE IF NOT EXISTS coredump(
                  step INTEGER,
                  stage TEXT,
                  data JSON);"""
        )

    @classmethod
    def set_process_group_info(cls, ranks_info):
        cls.ranks_info_ = ranks_info

    @classmethod
    def set_core(cls, model, optimizer, scheduler):
        cls.model_ = model
        cls.optimizer_ = optimizer
        cls.scheduler_ = scheduler

        if not isinstance(cls.model_, list):
            cls.model_ = [cls.model_]

        cls.initialize()

    @classmethod
    def should_dump(cls):
        return cls.step_ <= cls.dump_step_

    @classmethod
    def dump_main_grad(
        cls, param, param_name: str, stage_name: str = "main-grad-in-bwd"
    ):
        if not cls.should_dump():
            return

        param_info = {
            "name": param_name,
            "cksum": _get_cksum(param.main_grad),
            "shape": list(param.main_grad.shape),
            "type": str(param.main_grad.type()),
        }
        param_info.update(cls.ranks_info_)

        try:
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, json.dumps(param_info)),
            )
        except Exception as e:
            print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_main_param(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                main_param_exist = (
                    hasattr(param, "main_param") and param.main_param is not None
                )

                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param.main_param) if main_param_exist else None,
                    "shape": list(param.main_param.shape) if main_param_exist else None,
                    "type": str(param.main_param.type())
                    if hasattr(param, "main_param") and param.main_param is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_model(cls, stage_name: str):
        if not cls.should_dump():
            return

        for model in cls.model_:
            for name, param in model.named_parameters():
                param_info = {
                    "name": name,
                    "cksum": _get_cksum(param),
                    "shape": list(param.shape),
                    "type": str(param.type()),
                    "requires_grad": param.requires_grad,
                    "grad_cksum": _get_cksum(param.grad)
                    if param.grad is not None
                    else None,
                    "grad_shape": list(param.grad.shape)
                    if param.grad is not None
                    else None,
                    "grad_type": str(param.grad.type())
                    if param.grad is not None
                    else None,
                }
                param_info.update(cls.ranks_info_)
                try:
                    cls.db_.execute(
                        "INSERT INTO coredump VALUES (?, ?, ?);",
                        (cls.step_, stage_name, json.dumps(param_info)),
                    )
                except Exception as e:
                    print(f"Error inserting data into coredump: {e}")

    @classmethod
    def dump_schedule_table(cls, num_microbatches, num_model_chunks, 
                           microbatch_group_size_per_vp_stage, schedule_table, order_data=None, stage_name="schedule-table"):
        """Dump virtual pipeline schedule table information to coredump table."""
        if not cls.should_dump():
            return
            
        try:
            # Serialize schedule table data
            schedule_info = {
                "type": "schedule_table",
                "num_microbatches": num_microbatches,
                "num_model_chunks": num_model_chunks,
                "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
                "schedule_table": schedule_table,
                "schedule_length": len(schedule_table) if schedule_table else 0,
                "order_data": order_data,
                "order_length": len(order_data) if order_data else 0,
            }
            
            # Add rank information
            schedule_info.update(cls.ranks_info_)
            
            cls.db_.execute(
                "INSERT INTO coredump VALUES (?, ?, ?);",
                (cls.step_, stage_name, json.dumps(schedule_info)),
            )
            
        except Exception as e:
            print(f"Error inserting schedule table data into coredump: {e}")

    @classmethod
    def register_layernorm_grad_hooks(cls):
        if not cls.should_dump():
            return
        
        try:
            from megatron.core import parallel_state
        except ImportError:
            print("Warning: Cannot import parallel_state, skipping layernorm grad hook registration")
            return
        
        # ???????????
        cls.layernorm_grads_cache = {}
        
        def create_grad_hook(param_name):
            def grad_hook(grad):
                try:
                    # ??????????????
                    cls.layernorm_grads_cache[param_name] = {
                        "grad_norm": grad.norm().item(),
                        "grad_mean": grad.mean().item(), 
                        "grad_std": grad.std().item(),
                        "grad_max": grad.max().item(),
                        "grad_min": grad.min().item(),
                        "grad_cksum": _get_cksum(grad),
                        "grad_shape": list(grad.shape),
                        "grad_type": str(grad.type()),
                    }
                    
                except Exception as e:
                    print(f"Error in grad hook for {param_name}: {e}")
                return grad  # ?????????
            return grad_hook
        
        # ?????ayerNorm??????hook
        for model in cls.model_:
            for name, param in model.named_parameters():
                if ('layernorm' in name.lower() or 'rmsnorm' in name.lower()) and param.requires_grad:
                    # ??????hook
                    param.register_hook(create_grad_hook(name))
                    print(f"Registered grad hook for: {name}")

    @classmethod
    def dump_layernorm_grads(cls, stage_name: str):
        if not cls.should_dump():
            return
        
        try:
            from megatron.core import parallel_state
        except ImportError:
            print("Warning: Cannot import parallel_state, skipping layernorm grad dump")
            return
        
        # ???????????????????
        if not hasattr(cls, 'layernorm_grads_cache'):
            print(f"No layernorm grad cache found for stage {stage_name}")
            return
        
        # ??????????????ook????????
        for model in cls.model_:
            for name, param in model.named_parameters():
                if ('layernorm' in name.lower() or 'rmsnorm' in name.lower()) and param.requires_grad:
                    
                    grad_info = {
                        "name": name,
                        "stage": stage_name,
                        "param_type": "layernorm",
                        
                        # ?????????
                        "tp_size": parallel_state.get_tensor_model_parallel_world_size(),
                        "cp_size": parallel_state.get_context_parallel_world_size(),
                        "dp_size": parallel_state.get_data_parallel_world_size(),
                        "tp_rank": parallel_state.get_tensor_model_parallel_rank(),
                        "cp_rank": parallel_state.get_context_parallel_rank(),
                        "dp_rank": parallel_state.get_data_parallel_rank(),
                        
                        # ?????????
                        "param_shape": list(param.shape),
                        "param_cksum": _get_cksum(param),
                        
                        # ??????????
                        "sequence_parallel": getattr(param, "sequence_parallel", False),
                        "average_gradients_across_tp_domain": getattr(param, "average_gradients_across_tp_domain", False),
                    }
                    
                    # ??ook??????????????
                    if name in cls.layernorm_grads_cache:
                        grad_info.update(cls.layernorm_grads_cache[name])
                        grad_info["grad_exists"] = True
                        grad_info["grad_source"] = "hook_cache"
                    else:
                        grad_info["grad_exists"] = False
                        grad_info["grad_source"] = "no_cache"
                    
                    # ?????????????
                    grad_info["current_grad_exists"] = param.grad is not None
                    if hasattr(param, 'main_grad'):
                        grad_info["main_grad_exists"] = param.main_grad is not None
                    
                    grad_info.update(cls.ranks_info_)
                    
                    try:
                        cls.db_.execute(
                            "INSERT INTO coredump VALUES (?, ?, ?);",
                            (cls.step_, stage_name, json.dumps(grad_info)),
                        )
                    except Exception as e:
                        print(f"Error inserting layernorm grad data: {e}")

    @classmethod
    def step(cls):
        cls.step_ += 1
