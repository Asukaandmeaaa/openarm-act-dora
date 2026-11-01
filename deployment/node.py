import sys


def main():
    role=sys.argv[1]; args=sys.argv[2:]; sys.argv=[sys.argv[0]]
    if role=='simulation':
        from .simulation import main
        main()
    elif role=='robot':
        from .native_robot import main
        main(args[0])
    elif role=='record':
        from .record import main
        main()
    elif role=='camera':
        from .camera import main
        main(args[0])
    elif role=='policy':
        from .policy import main
        main()
    elif role=='control':
        from .control import main
        main(hardware=args==['hardware'])
    elif role=='observer':
        from dora_openarm_observer.main import main
        main()
    elif role=='client':
        from .chunk_execution import execution_settings
        if execution_settings()['mode']=='chunk100_exec50':
            from .fifty_client import main
            main()
        else:
            from dora_openarm_local_policy_server.main import main
            main()
    elif role=='executor':
        from dora_openarm_actions_executor import main as executor
        from .common import Log
        import dora
        from .feedforward import ReferenceVelocity
        from .chunk_execution import execution_settings
        execution=execution_settings()
        log=Log('executor'); Node=dora.Node
        class LoggedNode:
            def __init__(self): self.node=Node(); self.velocity=ReferenceVelocity()
            def __getattr__(self,name): return getattr(self.node,name)
            def send_output(self,name,value,metadata=None):
                if name in ('chunk_done', 'chunk_prefetch'):
                    return self.node.send_output(name,value,metadata)
                metadata=self.velocity.attach(name,value.field('qpos')[0].as_py(),metadata)
                log('executor_output',output=name,qpos=value.field('qpos')[0].as_py(),
                    metadata=metadata,provenance=(f"LeRobot ACT temporal ensemble; commit {execution['n_action_steps']} targets; 30Hz target hold + official 250Hz lowpass" if execution['mode']=='act_temporal' else (f"predict {execution.get('chunk_size',50)}, execute first {execution['n_action_steps']}; client tail blend new_weight={metadata.get('tail_blend_new_weight', 1.0)}; official Hermite + lowpass" if execution['mode']=='chunk100_exec50' else execution['mode']+' experiment; official Hermite + lowpass' if execution['mode']!='official_blend' else 'official recursive tail blend + Hermite + lowpass; timestamp identifies latest input, not every contributing old tail')))
                return self.node.send_output(name,value,metadata)
        if execution['mode']=='act_temporal':
            from .temporal_execution import run, make_controller
            # Import/warm the CPU ensemble BEFORE connecting this Dora node,
            # so the graph cannot start queries while Torch is still loading.
            controller=make_controller(execution, log)
            run(LoggedNode(),execution,log,controller=controller)
            return
        if execution['mode'] in ('chunked','chunked_prefetch','chunk100_exec50'):
            from .chunk_executor import main as run_chunked
            run_chunked(LoggedNode(),execution['n_action_steps'],log,
                        prefetch_ms=execution.get('prefetch_ms',0),
                        chunk_size=execution.get('chunk_size',50),
                        mode_name=execution['mode'] if execution['mode']=='chunk100_exec50' else None)
            return
        if execution['mode']=='pipelined':
            from .pipeline_execution import main as run_pipeline
            run_pipeline(LoggedNode(),execution,log)
            return
        executor.dora.Node=LoggedNode
        sys.argv += ['--arms','right,left','--upsample','--filter']
        executor.main()
    else: raise ValueError(role)


if __name__=='__main__': main()
