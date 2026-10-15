# Variable usage audit

Roles: **HV** = drives 2D HV; **CONS** = constraint threshold; **NL** = used by a nonlinear effect; **FC_S1** = Stage-1 FC selection; **LIN_ONLY** = only enters via linear weighted mean.

## IDS (42 metrics)

- HV-driving: **2**  |  Constraint: **8**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **22**

| Metric | Roles |
|---|---|
| Latency p99 (ms) | CONS,HV,NL |
| Success probability | CONS |
| Quality (0-1) | CONS |
| Jitter (ms) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per task (J) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Mass (kg) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Reduction ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Bandwidth (Mbps) | LIN_ONLY |
| DTN delay (s) | LIN_ONLY |
| Tier clearance | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per task (USD) | CONS,HV |
| NRE (kUSD) | LIN_ONLY |
| Ops hours/day | LIN_ONLY |
| Autonomy compat | CONS |
| PQC readiness | LIN_ONLY |
| Ops minutes / 1000 tasks | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`22` of `42`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Bandwidth (Mbps)
  Compute availability
  DTN delay (s)
  DTN enabled
  Data sensitivity
  ECC level
  Isolation support
  Jitter (ms)
  Mass (kg)
  NRE (kUSD)
  Offload ratio
  Ops hours/day
  Ops minutes / 1000 tasks
  PQC readiness
  Power margin (W)
  Queue depth (tasks)
  Recovery time (min)
  Reduction ratio
  Storage (GB)
  TX power (W)
  Tier clearance
  Volume (cm3)
```

## SUNCATCHER (41 metrics)

- HV-driving: **2**  |  Constraint: **4**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **24**

| Metric | Roles |
|---|---|
| Latency p99 (ms) | HV,NL |
| Success probability | LIN_ONLY |
| Quality score | LIN_ONLY |
| Jitter (ms) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per unit (J) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Reduction ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Bandwidth (Mbps) | LIN_ONLY |
| DTN delay (s) | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per unit (USD) | CONS,HV |
| NRE (MUSD) | LIN_ONLY |
| Ops min. / 1000 units | LIN_ONLY |
| Autonomy compat | LIN_ONLY |
| PQC readiness | LIN_ONLY |
| Compute mass (kg) | LIN_ONLY |
| Orbital altitude (km) | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`24` of `41`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Autonomy compat
  Bandwidth (Mbps)
  Compute availability
  Compute mass (kg)
  DTN delay (s)
  DTN enabled
  Data sensitivity
  ECC level
  Isolation support
  Jitter (ms)
  NRE (MUSD)
  Offload ratio
  Ops min. / 1000 units
  Orbital altitude (km)
  PQC readiness
  Power margin (W)
  Quality score
  Queue depth (tasks)
  Recovery time (min)
  Reduction ratio
  Storage (GB)
  Success probability
  TX power (W)
  Volume (cm3)
```

## JWST (41 metrics)

- HV-driving: **2**  |  Constraint: **6**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **22**

| Metric | Roles |
|---|---|
| Latency (hrs) | CONS,HV,NL |
| Success probability | CONS |
| Science quality (0-1) | CONS |
| Jitter (hrs) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per obs (kJ) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Mass (kg) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Compression ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Downlink rate (Mbps) | LIN_ONLY |
| DTN delay (hrs) | LIN_ONLY |
| Tier clearance | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per obs (USD) | HV |
| NRE (MUSD) | LIN_ONLY |
| Ops hrs/day | LIN_ONLY |
| Autonomy compat | LIN_ONLY |
| PQC readiness | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`22` of `41`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Autonomy compat
  Compression ratio
  Compute availability
  DTN delay (hrs)
  DTN enabled
  Data sensitivity
  Downlink rate (Mbps)
  ECC level
  Isolation support
  Jitter (hrs)
  Mass (kg)
  NRE (MUSD)
  Offload ratio
  Ops hrs/day
  PQC readiness
  Power margin (W)
  Queue depth (tasks)
  Recovery time (min)
  Storage (GB)
  TX power (W)
  Tier clearance
  Volume (cm3)
```

## RST (41 metrics)

- HV-driving: **2**  |  Constraint: **6**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **22**

| Metric | Roles |
|---|---|
| Latency (hrs) | CONS,HV,NL |
| Success probability | CONS |
| Science quality (0-1) | CONS |
| Jitter (hrs) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per obs (kJ) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Mass (kg) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Compression ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Downlink rate (Mbps) | LIN_ONLY |
| DTN delay (hrs) | LIN_ONLY |
| Tier clearance | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per obs (USD) | HV |
| NRE (MUSD) | LIN_ONLY |
| Ops hrs/day | LIN_ONLY |
| Autonomy compat | LIN_ONLY |
| PQC readiness | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`22` of `41`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Autonomy compat
  Compression ratio
  Compute availability
  DTN delay (hrs)
  DTN enabled
  Data sensitivity
  Downlink rate (Mbps)
  ECC level
  Isolation support
  Jitter (hrs)
  Mass (kg)
  NRE (MUSD)
  Offload ratio
  Ops hrs/day
  PQC readiness
  Power margin (W)
  Queue depth (tasks)
  Recovery time (min)
  Storage (GB)
  TX power (W)
  Tier clearance
  Volume (cm3)
```

## LEO_OOS (42 metrics)

- HV-driving: **2**  |  Constraint: **8**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **22**

| Metric | Roles |
|---|---|
| Latency p99 (ms) | CONS,HV,NL |
| Success probability | CONS |
| Quality (0-1) | CONS |
| Jitter (ms) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per task (J) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Mass (kg) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Reduction ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Bandwidth (Mbps) | LIN_ONLY |
| DTN delay (s) | LIN_ONLY |
| Tier clearance | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per task (USD) | CONS,HV |
| NRE (kUSD) | LIN_ONLY |
| Ops hours/day | LIN_ONLY |
| Autonomy compat | CONS |
| PQC readiness | LIN_ONLY |
| Ops minutes / 1000 tasks | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`22` of `42`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Bandwidth (Mbps)
  Compute availability
  DTN delay (s)
  DTN enabled
  Data sensitivity
  ECC level
  Isolation support
  Jitter (ms)
  Mass (kg)
  NRE (kUSD)
  Offload ratio
  Ops hours/day
  Ops minutes / 1000 tasks
  PQC readiness
  Power margin (W)
  Queue depth (tasks)
  Recovery time (min)
  Reduction ratio
  Storage (GB)
  TX power (W)
  Tier clearance
  Volume (cm3)
```

## DEEP_SPACE_CRUISE (41 metrics)

- HV-driving: **2**  |  Constraint: **6**  |  Nonlinear: **9**  |  FC Stage-1: **8**  |  Linear-only: **22**

| Metric | Roles |
|---|---|
| Latency (hrs) | CONS,HV,NL |
| Success probability | CONS |
| Science quality (0-1) | CONS |
| Jitter (hrs) | LIN_ONLY |
| Throughput (GFLOPS) | FC_S1,NL |
| Energy per obs (kJ) | NL |
| Peak power (W) | NL |
| Power margin (W) | LIN_ONLY |
| Mass (kg) | LIN_ONLY |
| Thermal margin (C) | NL |
| Link availability | NL |
| Contact duty cycle | NL |
| Compression ratio | LIN_ONLY |
| TX power (W) | LIN_ONLY |
| Downlink rate (Mbps) | LIN_ONLY |
| DTN delay (hrs) | LIN_ONLY |
| Tier clearance | LIN_ONLY |
| Security protocol level | NL |
| Key mgmt feasibility | CONS |
| TRL | CONS,FC_S1 |
| Rad dose margin (krad) | CONS,FC_S1 |
| SDC rate (per hr) | FC_S1,NL |
| Compute availability | LIN_ONLY |
| Recovery time (min) | LIN_ONLY |
| Cost per obs (USD) | HV |
| NRE (MUSD) | LIN_ONLY |
| Ops hrs/day | LIN_ONLY |
| Autonomy compat | LIN_ONLY |
| PQC readiness | LIN_ONLY |
| Volume (cm3) | LIN_ONLY |
| Offload ratio | LIN_ONLY |
| DTN enabled | LIN_ONLY |
| Queue depth (tasks) | LIN_ONLY |
| Data sensitivity | LIN_ONLY |
| Isolation support | LIN_ONLY |
| Storage (GB) | LIN_ONLY |
| ECC level | LIN_ONLY |
| FC_proc_type | FC_S1 |
| FC_bus_type | FC_S1 |
| FC_reconfigurable | FC_S1 |
| FC_mem_type | FC_S1 |

**Linear-only metrics (`22` of `41`)** — only enter the optimization through the weighted mean; never gate feasibility, never appear in the HV projection, never trigger a nonlinear effect. An ablation tests whether they matter.

```
  Autonomy compat
  Compression ratio
  Compute availability
  DTN delay (hrs)
  DTN enabled
  Data sensitivity
  Downlink rate (Mbps)
  ECC level
  Isolation support
  Jitter (hrs)
  Mass (kg)
  NRE (MUSD)
  Offload ratio
  Ops hrs/day
  PQC readiness
  Power margin (W)
  Queue depth (tasks)
  Recovery time (min)
  Storage (GB)
  TX power (W)
  Tier clearance
  Volume (cm3)
```
