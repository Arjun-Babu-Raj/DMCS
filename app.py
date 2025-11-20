# ==========================================
# IMPORT LIBRARIES
# ==========================================
# Streamlit: Web framework for building interactive dashboards
# NumPy: Numerical computing for matrix operations and arrays
# Pandas: Data manipulation and analysis
# Plotly: Interactive visualization library
# Copy: Deep copy utilities (imported but not actively used in this version)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import copy

# ==========================================
# SECTION 1: BACKGROUND PARAMETERS (ENGINE ROOM)
# ==========================================
# This section defines the biological and economic "laws" of the simulation.
# All costs are in Indian Rupees (INR) and represent 2019-2020 estimates.
# Probabilities are annual transition risks derived from longitudinal studies.

BASE_PARAMS = {
    # --- RETINOPATHY ---
    # Sources: Rachapelle et al. 2013, Polack et al. 2015
    # Diabetic retinopathy is a progressive eye disease affecting blood vessels in the retina
    'retino_trans': {
        'npdr_pdr': 0.059,    # Progression: Non-Proliferative -> Proliferative (5.9% annual risk)
        'npdr_edema': 0.060,  # Progression: Non-Proliferative -> Macular Edema (6% annual risk)
        'pdr_blind': 0.020,   # Progression: PDR -> Blindness (2% with laser treatment)
        'edema_blind': 0.030  # Progression: Edema -> Blindness (3% annual risk)
    },
    'retino_cost': {
        'followup': 338,      # Unit cost of outpatient ophthalmology visit (₹338)
        'laser': 14704,       # Cost of Laser Photocoagulation treatment (₹14,704)
        'blind_mgmt': 3595    # Annual cost of managing blindness/rehabilitation (₹3,595)
    },
    'retino_util': {
        'npdr': 0.71,         # Utility: Early stage retinopathy (0.71 on 0-1 scale where 1=perfect health)
        'pdr': 0.71,          # Utility: Proliferative stage
        'edema': 0.66,        # Utility: Macular Edema reduces vision quality
        'blind': 0.06         # Utility: Blindness (Significant QALY loss - near zero quality of life)
    },

    # --- NEPHROPATHY ---
    # Sources: UKPDS 64 (Adler et al. 2003), Rajapurkar et al. 2012
    # Diabetic nephropathy is kidney disease caused by diabetes
    'nephro_trans': {
        'micro_macro': 0.027, # Progression: Micro -> Macroalbuminuria (2.7% annual risk)
        'macro_esrd': 0.022,  # Progression: Macro -> End Stage Renal Disease (2.2% annual risk)
        'esrd_death': 0.163   # Annual mortality on Dialysis (16.3% - high due to limited access in India)
    },
    'nephro_cost': {
        'micro_mgmt': 477,    # Annual drug cost for microalbuminuria (ACE inhibitors, ₹477)
        'macro_mgmt': 477,    # Annual drug cost for macroalbuminuria
        'dialysis': 185016,   # Annual Hemodialysis cost (~156 sessions × ₹1186/session)
        'transplant': 85000   # One-time Renal Transplant cost (Public sector subsidized, ₹85,000)
    },
    'nephro_util': {
        'micro': 0.79,        # Asymptomatic stage (similar to uncomplicated diabetes)
        'macro': 0.64,        # Symptomatic stage with heavy burden
        'dialysis': 0.51,     # Dialysis significantly affects QALY (Source: Critselis et al. 2017)
        'transplant': 0.602   # Post-transplant utility (improved but not perfect)
    },

    # --- FOOT ULCER ---
    # Sources: Cheng et al. 2017, Prinja et al. 2016
    # Diabetic foot complications arise from nerve damage and poor circulation
    'foot_trans': {
        'ulcer_inf': 0.090,     # Progression: Simple Ulcer -> Infected (9% annual risk)
        'inf_amp_minor': 0.161, # Progression: Infected -> Minor Amputation (16.1% annual risk)
        'inf_amp_major': 0.049  # Progression: Infected -> Major Amputation (4.9% annual risk)
    },
    'foot_cost': {
        'ulcer_tx': 1150,       # Conservative management (dressings/antibiotics, ₹1,150)
        'infected_tx': 5274,    # Management of infection requiring hospitalization (₹5,274)
        'amputation': 15588     # Surgical amputation cost (₹15,588)
    },
    'foot_util': {
        'ulcer': 0.61,          # Pain and mobility issues from ulcer
        'infected': 0.57,       # Infection impact on quality of life
        'amp_minor': 0.56,      # Loss of toes - moderate disability
        'amp_major': 0.51       # Loss of limb - high disutility
    },

    # --- STROKE (Cerebrovascular Disease) ---
    # Sources: UKPDS 35, PMJAY Health Benefit Packages
    # Stroke risk is elevated in diabetic patients due to vascular damage
    'stroke_trans': {
        'risk': 0.0027,         # Annual risk of Stroke in Diabetes (0.27% from UKPDS)
        'acute_death': 0.20,    # 30-day mortality for acute stroke (20%)
        'post_death_add': 0.05  # Additional annual mortality risk post-stroke (5% per year)
    },
    'stroke_cost': {
        'acute': 100000,        # Acute management (Thrombolysis/Hospitalization, ₹100,000)
        'followup': 510         # Annual rehabilitation and medication (₹510)
    },
    'stroke_util': {
        'acute': 0.50,          # Acute phase utility (severe disability)
        'post': 0.667           # Post-stroke disability utility (partial recovery)
    },

    # --- CORONARY HEART DISEASE (CHD) ---
    # Sources: UKPDS 35, CREATE Registry (India)
    # Myocardial Infarction (heart attack) risk in diabetic population
    'chd_trans': {
        'risk': 0.011,          # Annual risk of MI - Myocardial Infarction (1.1% per year)
        'acute_death': 0.089,   # Acute MI mortality (8.9% from CREATE Registry India)
        'post_death_add': 0.04  # Increased mortality post-MI (4% additional annual risk)
    },
    'chd_cost': {
        'acute_med': 28744,     # Medical management of MI (₹28,744)
        'acute_surg': 37428,    # Angioplasty/Intervention weighted cost (₹37,428)
        'followup': 182         # Annual cardiac follow-up (₹182)
    },
    'chd_util': {
        'acute': 0.60,          # Utility during acute event (significant impairment)
        'post': 0.70            # Post-MI utility (moderate impairment)
    }
}

# ==========================================
# SECTION 2: MARKOV MODEL ENGINE
# ==========================================
# This class handles the core logic of the simulation. It tracks the movement 
# of the population cohort through different health states over time using
# a Markov Chain approach.

class MarkovModule:
    """
    Base class for Markov disease progression models.
    
    A Markov model simulates disease progression by dividing the population into 
    mutually exclusive health states and calculating transitions between states
    at discrete time intervals (annually in this model).
    
    Key Concepts:
    - States: Health conditions (e.g., "No Disease", "Mild", "Severe", "Death")
    - Transition Matrix (T): Probabilities of moving from one state to another
    - Trace: Population distribution across states over time
    - Discount Rate: Future costs and benefits are worth less than present ones
    """
    
    def __init__(self, params, rx_eff=0.0, sens=0.0, horizon=20, discount=0.03, cohort=100000):
        """
        Initialize the Markov model with simulation parameters.
        
        Args:
            params (dict): Disease-specific parameters (costs, utilities, transition probs)
            rx_eff (float): Treatment efficacy (0.0 = no effect, 1.0 = 100% reduction)
            sens (float): Test sensitivity (0.0 = detects nothing, 1.0 = detects all cases)
            horizon (int): Number of years to simulate
            discount (float): Annual discount rate for economic evaluation (typically 3%)
            cohort (int): Starting population size
        """
        self.params = params              # Store disease parameters
        self.rx_eff = rx_eff              # Treatment efficacy
        self.sens = sens                  # Test sensitivity
        self.horizon = horizon            # Time horizon in years
        self.discount = discount          # Discount rate (3% standard for India HTA)
        self.cohort = cohort              # Starting cohort size
        self.trace = None                 # Will store population matrix (time × states)
        self.costs = 0                    # Accumulated discounted costs
        self.disutilities = 0             # Accumulated disutilities (opposite of QALYs)
        self.outcomes = {}                # Clinical outcomes (e.g., # of strokes)
        self.state_distribution = {}      # Person-years spent in each state
        self.screenable_population = np.zeros(horizon)  # Tracks eligible population for screening

    def calc_econ(self, cost_vec, util_vec):
        """
        Calculate total discounted costs and QALYs based on state residency.
        
        This is the economic "engine" that converts health states into monetary
        and quality-of-life values.
        
        Args:
            cost_vec (np.array): Annual cost for each health state
            util_vec (np.array): Utility (quality of life) for each state (0-1 scale)
            
        Formula:
            Total Cost = Σ(Population in State × Cost of State × Discount Factor)
            QALYs = Σ(Population in State × Utility × Discount Factor)
        """
        # Create discount factor array: [1, 1/1.03, 1/1.03^2, ..., 1/1.03^20]
        disc = (1 + self.discount) ** -np.arange(self.horizon)
        
        # Total Cost = Sum across all states and years, applying discount
        # trace[t, s] = population in state s at time t
        # cost_vec[s] = annual cost of state s
        self.costs = np.sum(np.sum(self.trace * cost_vec, axis=1) * disc)
        
        # Calculate Disutilities (reduction from baseline health)
        # Base utility = 0.79 (uncomplicated diabetes, already below perfect health of 1.0)
        base_util = 0.79
        disutil_vec = np.maximum(0, base_util - util_vec)  # How much worse than baseline
        self.disutilities = np.sum(np.sum(self.trace * disutil_vec, axis=1) * disc)

    def record_distribution(self, state_names):
        """
        Store person-years spent in each health state for detailed reporting.
        
        Person-years = total years lived by all individuals in a given state
        Example: 1000 people in "Blindness" state for 10 years = 10,000 person-years
        
        Args:
            state_names (list): Names of health states in order
        """
        for idx, name in enumerate(state_names):
            # Sum across all time periods for this state
            self.state_distribution[name] = np.sum(self.trace[:, idx])

# ==========================================
# SECTION 2.1: DISEASE-SPECIFIC MARKOV MODULES
# ==========================================
# Each module defines a specific complication's health states (Markov "bubbles")
# and transition matrix (arrows between states). These are the "natural history"
# models that show how diseases progress over time.

class ModRetinopathy(MarkovModule):
    """
    Diabetic Retinopathy Progression Model
    
    Health States:
        0. No DR (Diabetic Retinopathy)
        1. NPDR (Non-Proliferative DR - early stage)
        2. PDR (Proliferative DR - severe, blood vessel growth)
        3. Macular Edema (fluid buildup, vision threat)
        4. Blindness (irreversible)
        5. Death
        
    Screening Effect: Reduces progression rates (NPDR→PDR, PDR→Blindness)
    """
    
    def run(self):
        # Define model structure
        n_s = 6  # Number of health states
        state_names = ["No DR", "NPDR (Stage 1)", "PDR (Severe)", "Macular Edema", "Blindness (End Stage)", "Death"]
        
        # Initialize trace matrix: rows = time, columns = states
        trace = np.zeros((self.horizon, n_s))
        trace[0, 0] = self.cohort  # Everyone starts with "No DR"
        
        # Extract parameters
        p = self.params['retino_trans']  # Transition probabilities
        # Calculate risk ratio using weighted average:
        # rr = (sensitivity * (1 - rx_eff)) + ((1 - sensitivity) * 1.0)
        # This represents: detected cases get treatment benefit, undetected cases get no benefit
        rr = (self.sens * (1 - self.rx_eff)) + ((1 - self.sens) * 1.0)
        p_dev = 0.02                     # Annual risk of developing NPDR from No DR (2%)
        p_d = 0.015                      # Background mortality (1.5% per year)
        
        # Build Transition Matrix (T)
        # Each row represents FROM state, each column represents TO state
        # All rows must sum to 1.0 (100% probability accounted for)
        T = np.zeros((n_s, n_s))
        
        # State 0: No DR → Can develop NPDR or die
        T[0,0] = 1 - (p_dev + p_d)  # Stay in No DR
        T[0,1] = p_dev               # Develop NPDR
        T[0,5] = p_d                 # Death
        
        # State 1: NPDR → Can progress to PDR, Edema, or die
        # SCREENING EFFECT APPLIED HERE: rr reduces progression risk
        T[1,1] = 1 - (p['npdr_pdr']*rr + p['npdr_edema']*rr + p_d)  # Stay NPDR
        T[1,2] = p['npdr_pdr'] * rr      # Progress to PDR (screening reduces this)
        T[1,3] = p['npdr_edema'] * rr    # Develop Edema (screening reduces this)
        T[1,5] = p_d                      # Death
        
        # State 2: PDR → Can progress to Blindness or die
        T[2,2] = 1 - (p['pdr_blind']*rr + p_d)  # Stay PDR
        T[2,4] = p['pdr_blind'] * rr             # Progress to Blindness (reduced by screening)
        T[2,5] = p_d                              # Death
        
        # State 3: Macular Edema → Can progress to Blindness or die
        T[3,3] = 1 - (p['edema_blind']*rr + p_d)  # Stay Edema
        T[3,4] = p['edema_blind'] * rr            # Progress to Blindness
        T[3,5] = p_d                               # Death
        
        # State 4: Blindness → Absorbing state (can only die)
        T[4,4] = 1 - p_d  # Stay blind
        T[4,5] = p_d      # Death
        
        # State 5: Death → Absorbing state
        T[5,5] = 1.0
        
        # Run Markov chain: multiply population vector by transition matrix each year
        for t in range(1, self.horizon):
            trace[t] = trace[t-1] @ T  # Matrix multiplication (@ operator)
        
        self.trace = trace
        
        # Calculate screenable population (exclude blind and dead)
        self.screenable_population = np.sum(trace[:, 0:4], axis=1)
        
        # Calculate economic outcomes
        c, u = self.params['retino_cost'], self.params['retino_util']
        cost_vector = np.array([0, c['followup'], c['laser'], c['laser'], c['blind_mgmt'], 0])
        util_vector = np.array([0.79, u['npdr'], u['pdr'], u['edema'], u['blind'], 0])
        self.calc_econ(cost_vector, util_vector)
        
        # Record clinical outcomes
        self.outcomes['Blindness (Years)'] = np.sum(trace[:, 4])  # Total person-years of blindness
        self.record_distribution(state_names)

class ModNephropathy(MarkovModule):
    """
    Diabetic Nephropathy (Kidney Disease) Progression Model
    
    Health States:
        0. No Nephropathy
        1. Micro-albuminuria (early kidney damage, protein in urine)
        2. Macro-albuminuria (advanced kidney damage)
        3. Dialysis/ESRD (End Stage Renal Disease - kidney failure)
        4. Death
        
    Screening Effect: Slows progression to macro-albuminuria and ESRD
    """
    
    def run(self):
        n_s = 5
        state_names = ["No Nephropathy", "Micro-albuminuria (Stage 1)", "Macro-albuminuria", "Dialysis (End Stage)", "Death"]
        trace = np.zeros((self.horizon, n_s))
        trace[0, 0] = self.cohort  # Everyone starts healthy
        
        p = self.params['nephro_trans']
        # Calculate risk ratio using weighted average:
        # rr = (sensitivity * (1 - rx_eff)) + ((1 - sensitivity) * 1.0)
        rr = (self.sens * (1 - self.rx_eff)) + ((1 - self.sens) * 1.0)
        p_d = 0.015           # Background mortality
        
        # Build Transition Matrix
        T = np.zeros((n_s, n_s))
        
        # State 0: No Nephropathy → Can develop Micro-albuminuria
        T[0,0] = 1 - (0.022 + p_d)  # Stay healthy (97.8%)
        T[0,1] = 0.022               # Develop Micro (2.2%)
        T[0,4] = p_d                 # Death
        
        # State 1: Micro-albuminuria → Can progress to Macro
        T[1,1] = 1 - (p['micro_macro']*rr + p_d)  # Stay Micro
        T[1,2] = p['micro_macro'] * rr             # Progress to Macro (screening reduces)
        T[1,4] = p_d                                # Death
        
        # State 2: Macro-albuminuria → Can progress to ESRD
        T[2,2] = 1 - (p['macro_esrd']*rr + p_d)  # Stay Macro
        T[2,3] = p['macro_esrd'] * rr             # Progress to ESRD (screening reduces)
        T[2,4] = p_d                               # Death
        
        # State 3: Dialysis/ESRD → High mortality rate (16.3%)
        T[3,3] = 1 - p['esrd_death']  # Stay on dialysis (83.7%)
        T[3,4] = p['esrd_death']       # Death (16.3% - very high!)
        
        # State 4: Death → Absorbing
        T[4,4] = 1.0
        
        # Run simulation
        for t in range(1, self.horizon):
            trace[t] = trace[t-1] @ T
        
        self.trace = trace
        self.screenable_population = np.sum(trace[:, 0:3], axis=1)  # Exclude dialysis/death
        
        # Calculate costs and utilities
        c, u = self.params['nephro_cost'], self.params['nephro_util']
        self.calc_econ(
            np.array([0, c['micro_mgmt'], c['macro_mgmt'], c['dialysis'], 0]),
            np.array([0.79, u['micro'], u['macro'], u['dialysis'], 0])
        )
        
        # Clinical outcomes
        self.outcomes['Dialysis (Years)'] = np.sum(trace[:, 3])
        self.outcomes['Deaths (Nephro Related)'] = np.sum(trace[:, 3] * p['esrd_death'])
        self.record_distribution(state_names)

class ModFoot(MarkovModule):
    """
    Diabetic Foot Complication Progression Model
    
    Health States:
        0. No Foot Issue
        1. Foot Ulcer (open wound, often painless due to neuropathy)
        2. Infected Ulcer (high amputation risk)
        3. Amputation (minor or major - irreversible disability)
        4. Death
        
    Screening Effect: Early detection prevents ulcer infection and amputation
    """
    
    def run(self):
        n_s = 5
        state_names = ["No Foot Issue", "Foot Ulcer (Stage 1)", "Infected Ulcer", "Amputation (End Stage)", "Death"]
        trace = np.zeros((self.horizon, n_s))
        trace[0, 0] = self.cohort
        
        p = self.params['foot_trans']
        # Calculate risk ratio using weighted average:
        # rr = (sensitivity * (1 - rx_eff)) + ((1 - sensitivity) * 1.0)
        rr = (self.sens * (1 - self.rx_eff)) + ((1 - self.sens) * 1.0)
        p_d = 0.015           # Background mortality
        
        # Build Transition Matrix
        T = np.zeros((n_s, n_s))
        
        # State 0: No Foot Issue → Can develop ulcer
        T[0,0] = 1 - (0.02 + p_d)  # Stay healthy (98%)
        T[0,1] = 0.02               # Develop ulcer (2%)
        T[0,4] = p_d                # Death
        
        # State 1: Foot Ulcer → Can become infected (screening helps wound care)
        T[1,1] = 1 - (p['ulcer_inf']*rr + p_d)  # Stay with ulcer
        T[1,2] = p['ulcer_inf'] * rr             # Infection (9% reduced by screening)
        T[1,4] = p_d                              # Death
        
        # State 2: Infected Ulcer → High amputation risk
        # Combine minor and major amputation probabilities
        p_amp = (p['inf_amp_minor'] + p['inf_amp_major']) * rr  # Total amp risk (~21%)
        T[2,2] = 1 - (p_amp + p_d)  # Stay infected
        T[2,3] = p_amp               # Amputation (screening reduces)
        T[2,4] = p_d                 # Death
        
        # State 3: Amputation → High mortality (10% per year)
        T[3,3] = 1 - 0.10  # Survive with amputation
        T[3,4] = 0.10       # Death (complications, infection)
        
        # State 4: Death → Absorbing
        T[4,4] = 1.0
        
        # Run simulation
        for t in range(1, self.horizon):
            trace[t] = trace[t-1] @ T
        
        self.trace = trace
        self.screenable_population = np.sum(trace[:, 0:3], axis=1)  # Exclude amputees/dead
        
        # Calculate costs and utilities
        c, u = self.params['foot_cost'], self.params['foot_util']
        self.calc_econ(
            np.array([0, c['ulcer_tx'], c['infected_tx'], c['amputation'], 0]),
            np.array([0.79, u['ulcer'], u['infected'], u['amp_major'], 0])
        )
        
        # Total amputations = current amputees + new deaths from amputation state
        self.outcomes['Amputations (Cases)'] = trace[-1, 3] + np.sum(trace[:, 3] * 0.10)
        self.record_distribution(state_names)

class ModStroke(MarkovModule):
    """
    Stroke (Cerebrovascular Accident) Model
    
    Health States:
        0. No Stroke (at risk population)
        1. Acute Stroke (emergency phase - high mortality)
        2. Post-Stroke (survived, but with disability)
        3. Death
        
    Screening Effect: Indirect - better diabetes control reduces vascular risk
    Note: Effectiveness is halved (0.5×) because screening has less direct impact
          compared to retinopathy/nephropathy where lesions are directly detected.
    """
    
    def run(self):
        n_s = 4
        state_names = ["No Stroke", "Acute Stroke", "Post Stroke History", "Death"]
        trace = np.zeros((self.horizon, n_s))
        trace[0, 0] = self.cohort
        
        p = self.params['stroke_trans']
        # Apply 50% efficacy factor to treatment, then calculate weighted risk ratio
        rr_treated = 1.0 - (self.rx_eff * 0.5)  # Reduced treatment effect for stroke
        # Calculate weighted rr: detected cases get reduced benefit, undetected get none
        rr = (self.sens * rr_treated) + ((1 - self.sens) * 1.0)
        p_event = p['risk'] * rr      # Annual stroke risk (0.27% reduced by screening)
        p_d_bg = 0.015                 # Background mortality
        
        # Build Transition Matrix
        T = np.zeros((n_s, n_s))
        
        # State 0: No Stroke → Can have stroke event
        T[0,0] = 1 - (p_event + p_d_bg)  # Stay stroke-free
        T[0,1] = p_event                  # Acute stroke event
        T[0,3] = p_d_bg                   # Death (other causes)
        
        # State 1: Acute Stroke → One-time state (transition immediately)
        T[1,2] = 1.0 - p['acute_death']  # Survive to post-stroke (80%)
        T[1,3] = p['acute_death']         # Death within 30 days (20%)
        
        # State 2: Post-Stroke → Elevated mortality risk
        p_d_post = p_d_bg + p['post_death_add']  # 1.5% + 5% = 6.5% annual mortality
        T[2,2] = 1.0 - p_d_post  # Survive with disability
        T[2,3] = p_d_post         # Death
        
        # State 3: Death → Absorbing
        T[3,3] = 1.0
        
        # Run simulation
        for t in range(1, self.horizon):
            trace[t] = trace[t-1] @ T
        
        self.trace = trace
        self.screenable_population = trace[:, 0]  # Only stroke-free are screened
        
        # Calculate costs and utilities
        c, u = self.params['stroke_cost'], self.params['stroke_util']
        c_vec = np.array([0, c['acute'], c['followup'], 0])
        u_vec = np.array([0.79, u['acute'], u['post'], 0])
        self.calc_econ(c_vec, u_vec)
        
        # Count total stroke events (person-years in acute state)
        self.outcomes['Stroke (Events)'] = np.sum(trace[:, 1])
        self.record_distribution(state_names)

class ModCHD(MarkovModule):
    """
    Coronary Heart Disease (CHD) / Myocardial Infarction (MI) Model
    
    Health States:
        0. No CHD (at risk population)
        1. Acute MI (heart attack - emergency)
        2. Post-MI (survived, chronic heart condition)
        3. Death
        
    Screening Effect: Indirect - similar to stroke, better diabetes control
                      reduces cardiovascular risk (50% effectiveness)
    """
    
    def run(self):
        n_s = 4
        state_names = ["No CHD", "Acute MI", "Post MI History", "Death"]
        trace = np.zeros((self.horizon, n_s))
        trace[0, 0] = self.cohort
        
        p = self.params['chd_trans']
        # Apply 50% efficacy factor to treatment, then calculate weighted risk ratio
        rr_treated = 1.0 - (self.rx_eff * 0.5)  # Reduced treatment effect for CHD
        # Calculate weighted rr: detected cases get reduced benefit, undetected get none
        rr = (self.sens * rr_treated) + ((1 - self.sens) * 1.0)
        p_event = p['risk'] * rr      # Annual MI risk (1.1% reduced by screening)
        p_d_bg = 0.015                 # Background mortality
        
        # Build Transition Matrix (identical structure to Stroke model)
        T = np.zeros((n_s, n_s))
        
        # State 0: No CHD → Can have MI event
        T[0,0] = 1 - (p_event + p_d_bg)  # Stay healthy
        T[0,1] = p_event                  # Acute MI
        T[0,3] = p_d_bg                   # Death (other causes)
        
        # State 1: Acute MI → One-time transition
        T[1,2] = 1.0 - p['acute_death']  # Survive to post-MI (91.1%)
        T[1,3] = p['acute_death']         # Death during acute phase (8.9%)
        
        # State 2: Post-MI → Elevated mortality
        p_d_post = p_d_bg + p['post_death_add']  # 1.5% + 4% = 5.5% annual mortality
        T[2,2] = 1.0 - p_d_post  # Live with CHD
        T[2,3] = p_d_post         # Death
        
        # State 3: Death → Absorbing
        T[3,3] = 1.0
        
        # Run simulation
        for t in range(1, self.horizon):
            trace[t] = trace[t-1] @ T
        
        self.trace = trace
        self.screenable_population = trace[:, 0]  # Only CHD-free are screened
        
        # Calculate costs and utilities
        c, u = self.params['chd_cost'], self.params['chd_util']
        # Average acute cost (medical vs surgical management)
        acute_cost = (c['acute_med'] + c['acute_surg']) / 2
        c_vec = np.array([0, acute_cost, c['followup'], 0])
        u_vec = np.array([0.79, u['acute'], u['post'], 0])
        self.calc_econ(c_vec, u_vec)
        
        # Count total MI events
        self.outcomes['Heart Attacks (MI)'] = np.sum(trace[:, 1])
        self.record_distribution(state_names)

# ==========================================
# SECTION 3: HELPER FUNCTIONS
# ==========================================

def perturb_dict(d, variability=0.2):
    """
    Recursively applies random noise to all parameters for sensitivity analysis.
    
    Probabilistic Sensitivity Analysis (PSA) tests model robustness by varying
    input parameters within plausible ranges. This function creates randomized
    parameter sets using uniform distribution.
    
    Args:
        d (dict): Nested dictionary of parameters (BASE_PARAMS structure)
        variability (float): Percentage variation (±20% default)
        
    Returns:
        dict: New parameter dictionary with randomized values
        
    Example:
        If cost = 100 and variability = 0.2:
        New cost will be randomly selected from [80, 120]
    """
    new_d = {}
    for k, v in d.items():
        new_sub_d = {}
        for sub_k, sub_v in v.items():
            # Apply uniform noise: value × [1-variability, 1+variability]
            noise = np.random.uniform(1 - variability, 1 + variability)
            val = sub_v * noise
            
            # Apply logical constraints:
            # - Probabilities and Utilities must be between 0 and 1
            # - Costs must be non-negative
            if 'util' in k or 'trans' in k:
                val = np.clip(val, 0.0, 1.0)  # Constrain to [0, 1]
            else:
                val = max(0.0, val)  # Costs can't be negative
                
            new_sub_d[sub_k] = val
        new_d[k] = new_sub_d
    return new_d

def run_ht_scenario(params_in, target, rx_eff_val, sens_val, spec_val, confirm_cost_val, unit_cost, f_mult, cohort_s, horizon, disc):
    """
    Master function to run the complete Health Technology Assessment (HTA) model.
    
    This orchestrates all disease modules and calculates the total economic impact
    of targeted screening interventions.
    
    Args:
        params_in (dict): Parameter dictionary (BASE_PARAMS or perturbed version)
        target (str): Which complication to screen for (e.g., "Retinopathy", "All")
        rx_eff_val (float): Treatment efficacy (0.0 to 1.0)
        sens_val (float): Test sensitivity (0.0 to 1.0)
        spec_val (float): Test specificity (0.0 to 1.0)
        confirm_cost_val (float): Cost of confirmatory test for false positives (₹)
        unit_cost (float): Cost per screening test (₹)
        f_mult (float): Frequency multiplier (1.0=annual, 2.0=twice yearly, etc.)
        cohort_s (int): Cohort size
        horizon (int): Time horizon in years
        disc (float): Discount rate
        
    Returns:
        dict: Comprehensive results including:
            - Treat_Cost: Treatment costs only
            - Screen_Cost: Screening costs only  
            - Total_Cost: Sum of treatment + screening
            - QALY: Quality-adjusted life years
            - Outcomes: Clinical events (strokes, amputations, etc.)
            - Distributions: Person-years in each health state
            - Total_Deaths: Total mortality
    """
    
    # STEP 1: Determine which modules get intervention effectiveness
    # Logic: Apply effectiveness ONLY to the targeted disease(s)
    e_ret, e_neph, e_foot, e_str, e_chd = 0.0, 0.0, 0.0, 0.0, 0.0
    s_ret, s_neph, s_foot, s_str, s_chd = 0.0, 0.0, 0.0, 0.0, 0.0
    
    if target == "Retinopathy" or target == "Population Screening (All)":
        e_ret = rx_eff_val
        s_ret = sens_val
    if target == "Nephropathy" or target == "Population Screening (All)":
        e_neph = rx_eff_val
        s_neph = sens_val
    if target == "Foot Ulcer" or target == "Population Screening (All)":
        e_foot = rx_eff_val
        s_foot = sens_val
    if target == "Stroke" or target == "Population Screening (All)":
        e_str = rx_eff_val
        s_str = sens_val
    if target == "CHD" or target == "Population Screening (All)":
        e_chd = rx_eff_val
        s_chd = sens_val
    
    # STEP 2: Initialize all 5 disease modules
    mods = [
        ModRetinopathy(params_in, e_ret, s_ret, horizon, disc, cohort_s),
        ModNephropathy(params_in, e_neph, s_neph, horizon, disc, cohort_s),
        ModFoot(params_in, e_foot, s_foot, horizon, disc, cohort_s),
        ModStroke(params_in, e_str, s_str, horizon, disc, cohort_s),
        ModCHD(params_in, e_chd, s_chd, horizon, disc, cohort_s)
    ]
    
    # STEP 3: Initialize accumulators
    total_treat_cost = 0   # Sum of all treatment costs
    total_screen_cost = 0  # Sum of all screening costs
    total_disutil = 0      # Sum of all disutilities (for QALY calculation)
    outcomes = {}          # Clinical outcomes dictionary
    state_dists = {}       # Health state distributions
    total_deaths = 0       # Total deaths across all modules
    
    # Create discount factor array for screening cost calculation
    disc_arr = (1 + disc) ** -np.arange(horizon)

    # STEP 4: Run each module and aggregate results
    for m in mods:
        m.run()  # Execute Markov simulation
        
        # Accumulate economic outputs
        total_treat_cost += m.costs
        total_disutil += m.disutilities
        
        # Merge outcomes and distributions
        outcomes.update(m.outcomes)
        state_dists.update(m.state_distribution)
        
        # Count deaths (last column of trace matrix at final time point)
        total_deaths += m.trace[-1, -1]
        
        # STEP 5: Calculate Screening Costs (only for targeted diseases)
        # Key Logic: Screening is applied to eligible population (not blind/dead/etc.)
        is_targeted = (target == "Population Screening (All)")
        
        # Check if this specific module is targeted
        if target == "Retinopathy" and isinstance(m, ModRetinopathy):
            is_targeted = True
        if target == "Nephropathy" and isinstance(m, ModNephropathy):
            is_targeted = True
        if target == "Foot Ulcer" and isinstance(m, ModFoot):
            is_targeted = True
        if target == "Stroke" and isinstance(m, ModStroke):
            is_targeted = True
        if target == "CHD" and isinstance(m, ModCHD):
            is_targeted = True
        
        if is_targeted:
            # Annual screening cost = eligible_pop × unit_cost × frequency × discount
            # Example: 100,000 people × ₹50/test × 2/year × 0.97 (discount) = ₹9.7M
            annual_screen_costs = m.screenable_population * (unit_cost * f_mult) * disc_arr
            total_screen_cost += np.sum(annual_screen_costs)
            
            # STEP 5a: Calculate False Positive Costs
            # False positives occur when test incorrectly identifies healthy people as having disease
            # False positive rate = 1 - specificity
            # Column 0 of trace is the healthy/susceptible state for each module
            false_positives = m.trace[:, 0] * (1 - spec_val)
            # Cost of confirmatory tests for false positives
            fp_costs = false_positives * confirm_cost_val * f_mult * disc_arr
            total_screen_cost += np.sum(fp_costs)
    
    # STEP 6: Calculate QALYs
    # Base QALY = cohort size × years × base utility (0.79 for diabetes)
    # Discounted over time horizon
    base_qaly = np.sum(np.full(horizon, cohort_s) * disc_arr) * 0.79
    # Actual QALY = Base - Disutilities (losses from complications)
    
    # STEP 7: Return comprehensive results
    return {
        'Treat_Cost': total_treat_cost,
        'Screen_Cost': total_screen_cost,
        'Total_Cost': total_treat_cost + total_screen_cost,
        'QALY': base_qaly - total_disutil,
        'Outcomes': outcomes,
        'Distributions': state_dists,
        'Total_Deaths': total_deaths
    }

# ==========================================
# SECTION 4: STREAMLIT USER INTERFACE
# ==========================================
# This section creates the interactive web dashboard using Streamlit.
# Users can adjust parameters, run simulations, and visualize results.

# Configure page settings
st.set_page_config(page_title="Targeted Screening HTA", layout="wide")
st.title("Diabetes Complications HTA Model for India")

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
# Session state persists data across reruns (when user changes inputs)
# This prevents losing results when users interact with the UI

if 'results' not in st.session_state:
    st.session_state.results = None      # Stores deterministic analysis results
if 'psa_results' not in st.session_state:
    st.session_state.psa_results = None  # Stores probabilistic sensitivity analysis results

# ==========================================
# SIDEBAR: INPUT CONTROLS
# ==========================================
# The sidebar contains all user-configurable parameters

st.sidebar.header("1. Select Target")
target_disease = st.sidebar.selectbox(
    "Target Complication",
    ["Retinopathy", "Nephropathy", "Foot Ulcer", "Stroke", "CHD", "Population Screening (All)"],
    help="Choose which diabetic complication to screen for. 'All' screens for all complications."
)

st.sidebar.divider()

st.sidebar.header("2. Intervention Specs")

# Screening Frequency Selector
freq_label = st.sidebar.selectbox(
    "Frequency", 
    ["Annually", "Twice Yearly", "Every 2 Years", "Every 3 Years", "Every 5 Years"], 
    index=0,
    help="How often screening is performed affects both cost and effectiveness"
)

# Map frequency labels to multipliers (times per year)
freq_map = {
    "Twice Yearly": 2.0,    # 2× per year = 2.0
    "Annually": 1.0,        # 1× per year = 1.0
    "Every 2 Years": 0.5,   # Once every 2 years = 0.5/year
    "Every 3 Years": 1/3,   # Once every 3 years = 0.33/year
    "Every 5 Years": 0.2    # Once every 5 years = 0.2/year
}
freq_mult = freq_map[freq_label]

# Screening Cost Input
screen_cost_unit = st.sidebar.number_input(
    "Unit Cost per Test (₹)", 
    value=50, 
    min_value=0,
    help="Cost of a single screening test (e.g., fundus photography = ₹50, HbA1c test = ₹200)"
)

# Calculate and display annualized cost
annual_cost = screen_cost_unit * freq_mult
st.sidebar.info(f"**Annualized Cost:** ₹{annual_cost:,.2f} per eligible person")

# Test Sensitivity Slider
sens_input = st.sidebar.slider(
    "Test Sensitivity (%)", 
    0, 100, 80,
    help="Percentage of true positives correctly identified by the test. 80% = test detects 80% of actual cases"
)
sensitivity = sens_input / 100.0  # Convert percentage to decimal (80% → 0.80)

# Test Specificity Slider
spec_input = st.sidebar.slider(
    "Test Specificity (%)", 
    0, 100, 90,
    help="Percentage of true negatives correctly identified by the test. 90% = test correctly identifies 90% of healthy individuals"
)
specificity = spec_input / 100.0  # Convert percentage to decimal (90% → 0.90)

# Treatment Efficacy Slider
rx_eff_input = st.sidebar.slider(
    "Treatment Efficacy (Risk Reduction %)", 
    0, 100, 30,
    help="Percentage reduction in disease progression risk when treatment is applied. 30% = treatment reduces progression by 30%"
)
rx_efficacy = rx_eff_input / 100.0  # Convert percentage to decimal (30% → 0.30)

# Confirmatory Test Cost Input
confirm_cost = st.sidebar.number_input(
    "Confirmatory Test Cost (₹)", 
    value=500, 
    min_value=0,
    help="Cost of confirmatory diagnostic test for false positives (e.g., detailed retinal exam, HbA1c confirmation)"
)

st.sidebar.divider()

st.sidebar.header("3. Cohort Settings")

# Population size
cohort_size = st.sidebar.number_input(
    "Cohort Size", 
    value=100000,
    help="Starting population size. 100,000 is typical for HTA models."
)

# Time horizon
time_horizon = st.sidebar.slider(
    "Time Horizon (Years)", 
    5, 30, 20,
    help="Number of years to simulate. 20 years is standard for chronic disease modeling."
)

# Discount rate
discount_rate = st.sidebar.slider(
    "Discount Rate (%)", 
    0, 10, 3,
    help="Annual discount rate for costs and QALYs. India HTA guidelines recommend 3%."
) / 100.0  # Convert to decimal

# ==========================================
# SENSITIVITY ANALYSIS SECTION (EXPANDABLE)
# ==========================================
with st.sidebar.expander("Sensitivity Analysis"):
    st.caption("Run a Probabilistic Sensitivity Analysis (PSA) to test model robustness against parameter uncertainty.")
    
    # Variability slider
    sa_variability = st.slider(
        "Uncertainty (+/- %)", 
        5, 50, 20, 5,
        help="Range of parameter variation. 20% = parameters vary by ±20% around base value"
    ) / 100.0
    
    # PSA Execution Button
    if st.button("Run PSA Simulation"):
        n_iter = 200  # Number of Monte Carlo iterations (fixed for performance)
        results = []
        progress_bar = st.progress(0)
        
        # Monte Carlo Loop
        for i in range(n_iter):
            # 1. Randomize all model parameters
            rnd_params = perturb_dict(BASE_PARAMS, variability=sa_variability)
            
            # 2. Randomize intervention parameters
            rnd_cost = screen_cost_unit * np.random.uniform(1 - sa_variability, 1 + sa_variability)
            rnd_rx_eff = np.clip(
                rx_efficacy * np.random.uniform(1 - sa_variability, 1 + sa_variability),
                0, 1.0  # Cap maximum treatment efficacy at 100%
            )
            rnd_sens = np.clip(
                sensitivity * np.random.uniform(1 - sa_variability, 1 + sa_variability),
                0, 1.0  # Cap maximum sensitivity at 100%
            )
            rnd_spec = np.clip(
                specificity * np.random.uniform(1 - sa_variability, 1 + sa_variability),
                0, 1.0  # Cap maximum specificity at 100%
            )
            rnd_confirm_cost = confirm_cost * np.random.uniform(1 - sa_variability, 1 + sa_variability)
            
            # 3. Run both scenarios (counterfactual and intervention)
            cf = run_ht_scenario(rnd_params, "None", 0.0, 0.0, 1.0, 0, 0, 0, cohort_size, time_horizon, discount_rate)
            interv = run_ht_scenario(rnd_params, target_disease, rnd_rx_eff, rnd_sens, rnd_spec, rnd_confirm_cost, rnd_cost, freq_mult, cohort_size, time_horizon, discount_rate)
            
            # 4. Store incremental results
            results.append({
                'dC': interv['Total_Cost'] - cf['Total_Cost'],  # Incremental Cost
                'dQ': interv['QALY'] - cf['QALY']                # Incremental QALYs
            })
            
            # Update progress bar
            progress_bar.progress((i + 1) / n_iter)
        
        # Store PSA results in session state
        st.session_state.psa_results = pd.DataFrame(results)

# ==========================================
# MAIN ANALYSIS EXECUTION
# ==========================================
# Primary action button to run deterministic (base case) analysis

if st.button("Run Analysis", type="primary"):
    # Run TWO scenarios for comparison:
    # 1. Counterfactual (no screening): Shows natural disease progression
    # 2. Intervention (with screening): Shows impact of screening program
    
    res_cf = run_ht_scenario(
        BASE_PARAMS,        # Use base parameters
        "None",             # No intervention
        0.0,                # Zero treatment efficacy
        0.0,                # Zero sensitivity
        1.0,                # Perfect specificity (no false positives in counterfactual)
        0,                  # Zero confirmatory test cost
        0,                  # Zero screening cost
        0,                  # Zero frequency
        cohort_size, 
        time_horizon, 
        discount_rate
    )
    
    res_int = run_ht_scenario(
        BASE_PARAMS,        # Use base parameters  
        target_disease,     # User-selected target
        rx_efficacy,        # User-selected treatment efficacy
        sensitivity,        # User-selected sensitivity
        specificity,        # User-selected specificity
        confirm_cost,       # User-selected confirmatory test cost
        screen_cost_unit,   # User-selected screening cost
        freq_mult,          # User-selected frequency
        cohort_size,
        time_horizon,
        discount_rate
    )
    
    # Store results in session state (persists across reruns)
    st.session_state.results = {
        'res_cf': res_cf,           # Counterfactual results
        'res_int': res_int,         # Intervention results
        'target': target_disease,   # Store target for display
        'freq': freq_label          # Store frequency for display
    }
    
    # Reset PSA results when new deterministic run is performed
    st.session_state.psa_results = None

# ==========================================
# DISPLAY DETERMINISTIC RESULTS
# ==========================================
# Show results if they exist in session state

if st.session_state.results:
    # Extract stored results
    res_cf = st.session_state.results['res_cf']
    res_int = st.session_state.results['res_int']
    
    # ==========================================
    # CALCULATE INCREMENTAL OUTCOMES (Δ = Intervention - Counterfactual)
    # ==========================================
    # These are the KEY metrics for HTA decision-making
    
    d_cost = res_int['Total_Cost'] - res_cf['Total_Cost']  # Incremental Cost
    d_qaly = res_int['QALY'] - res_cf['QALY']              # Incremental QALYs (health gain)
    d_deaths = res_cf['Total_Deaths'] - res_int['Total_Deaths']  # Deaths Averted
    
    # Calculate ICER (Incremental Cost-Effectiveness Ratio)
    # ICER = Additional Cost / Additional Benefit
    # This answers: "How much does it cost to gain 1 QALY?"
    icer = d_cost / d_qaly if d_qaly != 0 else 0
    
    # ==========================================
    # DISPLAY KEY METRICS (Top Row)
    # ==========================================
    st.subheader(f"Results for: {st.session_state.results['target']} Screening ({st.session_state.results['freq']})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Incremental Cost", 
            f"₹ {d_cost:,.0f}",
            delta_color="inverse",  # Red if cost increases (bad)
            help="Additional cost of screening program over 20 years"
        )
    
    with col2:
        st.metric(
            "QALYs Gained", 
            f"{d_qaly:,.0f}",
            help="Quality-Adjusted Life Years gained. 1 QALY = 1 year in perfect health"
        )
    
    with col3:
        st.metric(
            "Deaths Averted", 
            f"{d_deaths:,.0f}",
            delta_color="normal",  # Green if deaths reduced (good)
            help="Number of deaths prevented by screening intervention"
        )
    
    with col4:
        # ICER Interpretation (India-specific thresholds)
        # WHO Threshold: 1× GDP per capita ≈ ₹140,000/QALY (2020)
        if icer < 0:
            icer_label = "Dominant"  # Saves money AND improves health (rare!)
        elif icer < 140000:
            icer_label = "Cost Effective"  # Below willingness-to-pay threshold
        else:
            icer_label = "High Cost"  # Above threshold
        
        st.metric(
            "ICER (₹/QALY)", 
            f"₹ {icer:,.0f}",
            delta=icer_label,
            delta_color="inverse" if icer > 0 else "normal",
            help="Cost per QALY gained. ₹140,000 = India cost-effectiveness threshold (1× GDP per capita)"
        )

    st.divider()
    
    # ==========================================
    # DETAILED POPULATION DISTRIBUTION TABLE
    # ==========================================
    st.subheader("Disease Burden & Population Distribution")
    
    # Extract state distributions from both scenarios
    dist_cf = res_cf['Distributions']    # Counterfactual
    dist_int = res_int['Distributions']  # Intervention
    
    # ==========================================
    # FILTER RELEVANT STATES (based on target disease)
    # ==========================================
    relevant_keys = []
    
    if st.session_state.results['target'] == "Population Screening (All)":
        # Show all states if screening all complications
        relevant_keys = list(dist_cf.keys())
    else:
        # Filter states related to target disease using keyword matching
        keywords = {
            "Retinopathy": ["DR", "NPDR", "PDR", "Edema", "Blind", "Retino"],
            "Nephropathy": ["Neph", "Albuminuria", "Dialysis", "ESRD"],
            "Foot Ulcer": ["Foot", "Ulcer", "Amputation"],
            "Stroke": ["Stroke"],
            "CHD": ["CHD", "MI"]
        }
        target_keys = keywords.get(st.session_state.results['target'], [])
        
        for k in dist_cf.keys():
            # Case-insensitive matching for robustness
            if any(tk.lower() in k.lower() for tk in target_keys):
                relevant_keys.append(k)
    
    # ==========================================
    # BUILD DATA TABLE
    # ==========================================
    data_rows = []
    
    # Keywords to identify "bad" health outcomes (should decrease with screening)
    bad_outcomes = ["Blindness", "Dialysis", "Amputation", "Death", "Acute", "Severe"]
    
    for k in relevant_keys:
        val_cf = dist_cf[k]      # Person-years without screening
        val_int = dist_int[k]    # Person-years with screening
        diff = val_int - val_cf  # Change (negative for bad outcomes = good!)
        
        # Categorize health state
        status = "Neutral"
        if any(b in k for b in bad_outcomes):
            status = "Bad Outcome"
        elif "No " in k:
            status = "Healthy State"
        else:
            status = "Early Stage (Manageable)"
        
        data_rows.append({
            "Health State": k,
            "Category": status,
            "Person-Years (No Screen)": val_cf,
            "Person-Years (Screening)": val_int,
            "Difference (Years)": diff
        })
    
    df_dist = pd.DataFrame(data_rows)
    
    # ==========================================
    # COLOR CODING FUNCTION (Green = Good, Red = Bad)
    # ==========================================
    def highlight_rows(row):
        """
        Apply conditional formatting based on outcome direction.
        
        Logic:
        - Bad outcomes decreasing (diff < 0) → GREEN (good)
        - Healthy states increasing (diff > 0) → GREEN (good)
        - Bad outcomes increasing (diff > 0) → RED (bad)
        """
        diff = row["Difference (Years)"]
        cat = row["Category"]
        color = ''  # Default: no color
        
        # GREEN: Good changes
        if cat == "Bad Outcome" and diff < 0:
            color = 'background-color: #d4edda; color: #155724'  # Light green
        elif (cat == "Healthy State" or cat == "Early Stage (Manageable)") and diff > 0:
            color = 'background-color: #d4edda; color: #155724'
        
        # RED: Bad changes
        elif cat == "Bad Outcome" and diff > 0:
            color = 'background-color: #f8d7da; color: #721c24'  # Light red
        
        return [color] * len(row)

    # Display styled dataframe
    if not df_dist.empty:
        st.dataframe(
            df_dist.style.apply(highlight_rows, axis=1).format({
                "Person-Years (No Screen)": "{:,.0f}",
                "Person-Years (Screening)": "{:,.0f}",
                "Difference (Years)": "{:+,.0f}"  # + sign for positive numbers
            }),
            use_container_width=True,
            height=600
        )

        # ==========================================
        # VISUALIZATION: POPULATION SHIFT BAR CHART
        # ==========================================
        fig = go.Figure()
        
        # Add bar for No Screening scenario
        fig.add_trace(go.Bar(
            name='No Screening',
            x=df_dist['Health State'],
            y=df_dist['Person-Years (No Screen)'],
            marker_color='lightgrey'
        ))
        
        # Add bar for Screening scenario
        fig.add_trace(go.Bar(
            name='Screening',
            x=df_dist['Health State'],
            y=df_dist['Person-Years (Screening)'],
            marker_color='teal'
        ))
        
        fig.update_layout(
            barmode='group',           # Grouped bars side-by-side
            xaxis_tickangle=-45,       # Rotate labels for readability
            title="Population Shift by Health State",
            xaxis_title="Health State",
            yaxis_title="Person-Years",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# DISPLAY PROBABILISTIC SENSITIVITY ANALYSIS (PSA) RESULTS
# ==========================================
# Show PSA results if they exist in session state

if st.session_state.psa_results is not None:
    st.divider()
    st.subheader("Probabilistic Sensitivity Analysis")
    
    df_psa = st.session_state.psa_results
    
    # ==========================================
    # COST-EFFECTIVENESS PLANE (CE Plane)
    # ==========================================
    # This is the standard visualization for HTA uncertainty analysis
    # X-axis: Incremental QALYs (health benefit)
    # Y-axis: Incremental Cost (economic burden)
    # Quadrants:
    #   - Northeast (top-right): More costly, more effective → Need ICER analysis
    #   - Southeast (bottom-right): Less costly, more effective → DOMINANT (best!)
    #   - Northwest (top-left): More costly, less effective → DOMINATED (worst!)
    #   - Southwest (bottom-left): Less costly, less effective → Need ICER analysis
    
    fig = go.Figure()
    
    # Plot simulation points (Monte Carlo results)
    fig.add_trace(go.Scatter(
        x=df_psa['dQ'],           # Incremental QALYs
        y=df_psa['dC'],           # Incremental Cost
        mode='markers',
        marker=dict(
            color='teal',
            opacity=0.6,
            size=8
        ),
        name='Simulations',
        hovertemplate='<b>QALY:</b> %{x:,.0f}<br><b>Cost:</b> ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add Willingness-to-Pay (WTP) Threshold Line
    # WHO recommendation: 1× GDP per capita for cost-effective interventions
    # India GDP per capita (2020) ≈ ₹140,000
    wtp = 140000
    x_range = np.linspace(df_psa['dQ'].min(), df_psa['dQ'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=x_range * wtp,  # Linear relationship: Cost = WTP × QALYs
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'WTP Threshold (₹{wtp:,}/QALY)',
        hovertemplate='<b>Threshold ICER:</b> ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add reference lines (origin axes)
    fig.add_hline(y=0, line_width=1, line_color="black", line_dash="dot")
    fig.add_vline(x=0, line_width=1, line_color="black", line_dash="dot")
    
    fig.update_layout(
        title="Cost-Effectiveness Plane (Uncertainty Analysis)",
        xaxis_title="Incremental QALYs (Health Benefit)",
        yaxis_title="Incremental Cost (₹)",
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # PSA SUMMARY STATISTICS
    # ==========================================
    
    # Mean ICER across all simulations
    mean_icers = np.mean(df_psa['dC']) / np.mean(df_psa['dQ'])
    
    # Probability of Cost-Effectiveness
    # Calculate Net Monetary Benefit (NMB) = QALY_gained × WTP - Cost
    # Intervention is cost-effective if NMB > 0
    prob_ce = np.mean((df_psa['dQ'] * wtp - df_psa['dC']) > 0)
    
    # Display summary in info box
    st.info(
        f"""
        **PSA Results ({len(df_psa)} Iterations):**
        - **Mean ICER:** ₹ {mean_icers:,.0f} per QALY
        - **Probability Cost-Effective at ₹{wtp:,}:** {prob_ce:.1%}
        
        **Interpretation:**
        - {prob_ce:.0%} of simulations fall below the cost-effectiveness threshold
        - This indicates {('HIGH' if prob_ce > 0.8 else 'MODERATE' if prob_ce > 0.5 else 'LOW')} confidence in cost-effectiveness
        """
    )

# ==========================================
# PRICE & FREQUENCY OPTIMIZER (TWO-WAY SENSITIVITY ANALYSIS)
# ==========================================
# This section performs a comprehensive grid search to map the cost-effectiveness
# landscape across different combinations of unit costs and screening frequencies.

st.divider()
st.header("Price & Frequency Optimizer")
st.caption("Two-Way Sensitivity Analysis: Find the break-even point where screening becomes cost-effective")

# Initialize session state for optimizer results
if 'optimizer_results' not in st.session_state:
    st.session_state.optimizer_results = None

# Expandable section for optimizer controls
with st.expander("Optimizer Settings", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost-effectiveness threshold (Willingness-to-Pay)
        wtp_threshold = st.number_input(
            "Cost-Effectiveness Threshold (₹/QALY)",
            value=140000,
            min_value=50000,
            max_value=500000,
            step=10000,
            help="India standard: ₹140,000 (1× GDP per capita). Adjust based on policy context."
        )
        
        # Unit cost range
        st.write("**Unit Cost Range (₹)**")
        cost_min = st.number_input("Minimum", value=10, min_value=0, max_value=1000)
        cost_max = st.number_input("Maximum", value=500, min_value=10, max_value=2000)
        cost_step = st.number_input("Step Size", value=20, min_value=5, max_value=100)
    
    with col2:
        # Frequency options (using same mapping as main analysis)
        # Note: Excluding "Twice Yearly" as it's less commonly used for chronic disease screening
        freq_options = ["Annually", "Every 2 Years", "Every 3 Years", "Every 5 Years"]
        selected_freqs = st.multiselect(
            "Screening Frequencies to Test",
            freq_options,
            default=freq_options,
            help="Select which frequencies to include in the analysis"
        )
        
        st.info(
            """
            **How to Interpret Results:**
            
            🟢 **Green Zones:** Cost-effective at this threshold  
            🟡 **Yellow Zones:** Marginal cost-effectiveness  
            🔴 **Red Zones:** Not cost-effective (too expensive)
            
            **Actionable Insights:**
            - Identify maximum affordable price for each frequency
            - Find optimal frequency-price combinations
            - Understand trade-offs between cost and frequency
            """
        )

# Run Optimizer Button
if st.button("Run Price & Frequency Optimizer", type="primary"):
    if not selected_freqs:
        st.error("Please select at least one frequency to analyze.")
    else:
        # Generate cost range
        unit_costs = np.arange(cost_min, cost_max + cost_step, cost_step)
        
        # Map frequency labels to multipliers (times per year)
        freq_map = {
            "Annually": 1.0,
            "Every 2 Years": 0.5,
            "Every 3 Years": 1/3,
            "Every 5 Years": 0.2
        }
        
        # Initialize results matrix
        icer_matrix = []
        freq_labels = []
        
        # Progress tracking
        total_runs = len(selected_freqs) * len(unit_costs)
        current_run = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Grid search: iterate through all combinations
        for freq_label in selected_freqs:
            freq_mult = freq_map[freq_label]
            icer_row = []
            
            for unit_cost in unit_costs:
                # Update progress
                current_run += 1
                progress_bar.progress(current_run / total_runs)
                status_text.text(f"Running simulation {current_run}/{total_runs}: {freq_label} @ ₹{unit_cost:.0f}...")
                
                # Run counterfactual (no screening)
                res_cf = run_ht_scenario(
                    BASE_PARAMS,
                    "None",
                    0.0,
                    0.0,
                    1.0,
                    0,
                    0,
                    0,
                    cohort_size,
                    time_horizon,
                    discount_rate
                )
                
                # Run intervention with current cost/frequency combination
                res_int = run_ht_scenario(
                    BASE_PARAMS,
                    target_disease,
                    rx_efficacy,
                    sensitivity,
                    specificity,
                    confirm_cost,
                    unit_cost,
                    freq_mult,
                    cohort_size,
                    time_horizon,
                    discount_rate
                )
                
                # Calculate ICER
                d_cost = res_int['Total_Cost'] - res_cf['Total_Cost']
                d_qaly = res_int['QALY'] - res_cf['QALY']
                
                # ICER interpretation:
                # - d_qaly > 0, d_cost > 0: Standard case, ICER = cost/benefit
                # - d_qaly > 0, d_cost < 0: Dominant (saves money and improves health), negative ICER
                # - d_qaly < 0: Dominated (harms health), ICER = infinity
                if d_qaly > 0:
                    icer = d_cost / d_qaly  # Can be positive or negative
                else:
                    icer = np.inf  # Intervention is dominated (no benefit or causes harm)
                
                icer_row.append(icer)
            
            icer_matrix.append(icer_row)
            freq_labels.append(freq_label)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Store results in session state
        st.session_state.optimizer_results = {
            'icer_matrix': np.array(icer_matrix),
            'unit_costs': unit_costs,
            'freq_labels': freq_labels,
            'wtp_threshold': wtp_threshold,
            'target': target_disease
        }
        
        st.success(f"Completed {total_runs} simulations!")

# Display optimizer results if available
if st.session_state.optimizer_results is not None:
    results = st.session_state.optimizer_results
    icer_matrix = results['icer_matrix']
    unit_costs = results['unit_costs']
    freq_labels = results['freq_labels']
    wtp_threshold = results['wtp_threshold']
    
    st.subheader(f"Results: {results['target']} Screening")
    
    # ==========================================
    # CREATE HEATMAP VISUALIZATION
    # ==========================================
    
    # Constants for visualization
    ICER_DISPLAY_MULTIPLIER = 3  # Cap display at 3× threshold for better color scaling
    
    # Prepare data for heatmap
    # Cap ICERs at threshold multiplier for better visualization (extremely high values compress color scale)
    icer_display = np.minimum(icer_matrix, wtp_threshold * ICER_DISPLAY_MULTIPLIER)
    
    # Create custom colorscale
    # Green (0) → Yellow (threshold) → Red (3× threshold)
    colorscale = [
        [0.0, '#155724'],      # Dark green (dominant/highly cost-effective)
        [0.33, '#28a745'],     # Green (cost-effective)
        [0.5, '#ffc107'],      # Yellow (at threshold)
        [0.67, '#fd7e14'],     # Orange (marginally not cost-effective)
        [1.0, '#dc3545']       # Red (not cost-effective)
    ]
    
    # Create hover text with formatted ICER values
    hover_text = []
    for i, freq in enumerate(freq_labels):
        row_text = []
        for j, cost in enumerate(unit_costs):
            icer_val = icer_matrix[i, j]
            if np.isinf(icer_val):
                text = f"Frequency: {freq}<br>Unit Cost: ₹{cost:.0f}<br>ICER: Dominated (No QALY gain)"
            else:
                ce_status = "Cost-Effective ✓" if icer_val <= wtp_threshold else "Not Cost-Effective ✗"
                text = f"Frequency: {freq}<br>Unit Cost: ₹{cost:.0f}<br>ICER: ₹{icer_val:,.0f}/QALY<br>{ce_status}"
            row_text.append(text)
        hover_text.append(row_text)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=icer_display,
        x=unit_costs,
        y=freq_labels,
        colorscale=colorscale,
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        colorbar=dict(
            title="ICER<br>(₹/QALY)",
            tickformat=',.0f',
            len=0.7
        )
    ))
    
    # Add threshold annotation
    fig.add_annotation(
        text=f"Cost-Effectiveness Threshold: ₹{wtp_threshold:,}/QALY",
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12, color="red"),
        align="center"
    )
    
    fig.update_layout(
        title="Cost-Effectiveness Landscape: ICER by Unit Cost and Frequency",
        xaxis_title="Unit Cost per Test (₹)",
        yaxis_title="Screening Frequency",
        height=500,
        font=dict(size=11)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ==========================================
    # ACTIONABLE INSIGHTS TABLE
    # ==========================================
    
    st.subheader("Actionable Insights: Break-Even Analysis")
    
    insights_data = []
    
    for i, freq in enumerate(freq_labels):
        # Find maximum cost-effective price (only among finite ICERs)
        finite_mask = np.isfinite(icer_matrix[i, :])
        ce_mask = (icer_matrix[i, :] <= wtp_threshold) & finite_mask
        ce_costs = unit_costs[ce_mask]
        
        if len(ce_costs) > 0:
            max_ce_cost = ce_costs.max()
            
            # Find the cost that gives minimum finite ICER
            finite_icers_row = icer_matrix[i, finite_mask]
            finite_costs_row = unit_costs[finite_mask]
            
            if len(finite_icers_row) > 0:
                min_icer_idx = finite_icers_row.argmin()
                min_icer = finite_icers_row[min_icer_idx]
                optimal_cost = finite_costs_row[min_icer_idx]
                
                # Format ICER display
                icer_display = f"₹{min_icer:,.0f}" + (" (Dominant)" if min_icer < 0 else "")
                
                insights_data.append({
                    "Frequency": freq,
                    "Max Cost-Effective Price (₹)": f"₹{max_ce_cost:.0f}",
                    "Optimal Unit Cost (₹)": f"₹{optimal_cost:.0f}",
                    "Best ICER (₹/QALY)": icer_display,
                    "Recommendation": "Cost-Effective" if max_ce_cost >= cost_min else "Limited Range"
                })
            else:
                # All ICERs are infinite (dominated)
                insights_data.append({
                    "Frequency": freq,
                    "Max Cost-Effective Price (₹)": "N/A",
                    "Optimal Unit Cost (₹)": "N/A",
                    "Best ICER (₹/QALY)": "Dominated (No QALY gain)",
                    "Recommendation": "Not Viable (No Health Benefit)"
                })
        else:
            # No cost-effective options in this range
            finite_icers_row = icer_matrix[i, finite_mask]
            if len(finite_icers_row) > 0:
                min_icer = finite_icers_row.min()
                icer_display = f"₹{min_icer:,.0f}" + (" (Dominant)" if min_icer < 0 else "")
            else:
                icer_display = "Dominated (No QALY gain)"
            
            insights_data.append({
                "Frequency": freq,
                "Max Cost-Effective Price (₹)": f"< ₹{cost_min:.0f}",
                "Optimal Unit Cost (₹)": "N/A",
                "Best ICER (₹/QALY)": icer_display,
                "Recommendation": "Not Viable (Too Expensive)" if len(finite_icers_row) > 0 else "Not Viable (No Health Benefit)"
            })
    
    df_insights = pd.DataFrame(insights_data)
    st.dataframe(df_insights, use_container_width=True, hide_index=True)
    
    # ==========================================
    # KEY FINDINGS SUMMARY
    # ==========================================
    
    # Constants for classification
    HIGHLY_CE_MULTIPLIER = 0.5  # Threshold for "highly cost-effective" (50% of WTP)
    
    st.subheader("🎯 Key Findings & Policy Recommendations")
    
    # Find the most cost-effective combination (excluding dominated interventions)
    # Filter out infinite ICER values (dominated interventions where d_qaly <= 0)
    finite_icers = icer_matrix.copy()
    finite_icers[~np.isfinite(finite_icers)] = np.nan  # Replace inf with NaN for proper handling
    
    if np.all(np.isnan(finite_icers)):
        st.error("All simulated combinations are dominated (no QALY gain). Cannot recommend an optimal strategy.")
    else:
        min_icer_idx = np.unravel_index(np.nanargmin(finite_icers), finite_icers.shape)
        best_freq = freq_labels[min_icer_idx[0]]
        best_cost = unit_costs[min_icer_idx[1]]
        best_icer = icer_matrix[min_icer_idx]
        
        # Count how many combinations are cost-effective (exclude dominated interventions)
        total_combinations = icer_matrix.size
        finite_combinations = np.sum(np.isfinite(icer_matrix))
        dominated_combinations = total_combinations - finite_combinations
        ce_combinations = np.sum((icer_matrix <= wtp_threshold) & np.isfinite(icer_matrix))
        ce_percentage = (ce_combinations / total_combinations) * 100
        
        # Format ICER display for optimal strategy
        if best_icer < 0:
            icer_display = f"₹{best_icer:,.0f}/QALY (Dominant - Saves Money & Improves Health)"
            status = "Dominant (Best Possible Outcome)"
        elif best_icer < wtp_threshold * HIGHLY_CE_MULTIPLIER:
            icer_display = f"₹{best_icer:,.0f}/QALY"
            status = "Highly Cost-Effective"
        elif best_icer <= wtp_threshold:
            icer_display = f"₹{best_icer:,.0f}/QALY"
            status = "Cost-Effective"
        else:
            icer_display = f"₹{best_icer:,.0f}/QALY"
            status = "Above Threshold"
        
        st.success(
            f"""
            **Optimal Strategy:** {best_freq} screening at ₹{best_cost:.0f} per test
            - **ICER:** {icer_display}
            - **Status:** {status}
            
            **Overall Landscape:**
            - {ce_combinations}/{total_combinations} combinations ({ce_percentage:.0f}%) are cost-effective at ₹{wtp_threshold:,}/QALY threshold
            - {finite_combinations}/{total_combinations} combinations have QALY gains (non-dominated)
            """ + (f"\n-  {dominated_combinations} combinations are dominated (no health benefit)" if dominated_combinations > 0 else "")
        )
    
        # Generate policy recommendations
        st.markdown("### Policy Recommendations")
        
        # Find "green frontier" - transition points from cost-effective to not cost-effective
        for i, freq in enumerate(freq_labels):
            # Only consider finite ICERs (non-dominated interventions)
            finite_mask = np.isfinite(icer_matrix[i, :])
            ce_mask = (icer_matrix[i, :] <= wtp_threshold) & finite_mask
            ce_costs = unit_costs[ce_mask]
            
            if len(ce_costs) > 0:
                max_price = ce_costs.max()
                # Find minimum ICER for this frequency (among finite values)
                finite_icers_row = icer_matrix[i, finite_mask]
                if len(finite_icers_row) > 0:
                    min_icer_row = finite_icers_row.min()
                    icer_display_row = f"₹{min_icer_row:,.0f}/QALY" + (" (Dominant)" if min_icer_row < 0 else "")
                else:
                    icer_display_row = "N/A"
                
                st.markdown(
                    f"""
                    **{freq}:**  
                    - Cost-effective up to ₹{max_price:.0f} per test
                    - Best ICER: {icer_display_row}
                    -  **Action:** Negotiate test prices below ₹{max_price:.0f} to ensure viability
                    -  **Implication:** If market price exceeds ₹{max_price:.0f}, consider reducing frequency or providing subsidies
                    """
                )
            else:
                st.markdown(
                    f"""
                    **{freq}:**  
                    - Not cost-effective in tested range (₹{cost_min:.0f}-₹{cost_max:.0f})
                    -  **Action:** Either negotiate significantly lower prices (< ₹{cost_min:.0f}) or avoid this frequency
                    """
                )