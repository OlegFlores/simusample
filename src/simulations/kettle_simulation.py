# src/simulations/kettle_simulation.py
import simpy # Ensure this is at the top
import math
from pydantic import BaseModel, Field, conint, confloat, constr, validator
from enum import Enum
from typing import Optional

def thermal_mass_j_per_c(csa_mm2, length_m):
    c = 385  # J/kg·°C for copper
    density = 8960  # kg/m³ for copper
    area_m2 = csa_mm2 * 1e-6
    total_length = length_m * 2  # round trip
    volume = area_m2 * total_length  # m³
    mass = volume * density  # kg
    print(csa_mm2)
    print(length_m)
    print(c * mass)
    return c * mass  # J/°C

class BreakerTypeEnum(str, Enum):
    B = "B"
    C = "C"
    D = "D"

class LoadParameters(BaseModel):
    power_w: confloat(gt=0) = 0
    voltage_v: confloat(gt=0) # e.g., 120 or 230
    operation_time_s: confloat(ge=60, le=3600) # 1 min to 1 hour

class WireParameters(BaseModel):
    csa_mm2: confloat(gt=0) # e.g., 1.0, 1.5, 2.5, 4.0
    length_m: confloat(ge=1, le=50) # One-way length
    material_resistivity_ohm_mm2_per_m: confloat(gt=0) = 0.0175 # Copper
    # max_safe_current_a will be looked up, not an input here directly for the wire itself
    # It's a property of the wire type, often tied to installation method too.

class BreakerParameters(BaseModel):
    current_a: conint(gt=0) # e.g., 6, 10, 13, 16, 20, 25
    type: BreakerTypeEnum

class SimulationInput(BaseModel):
    load: LoadParameters
    wire: WireParameters
    breaker: BreakerParameters
    ambient_temperature_c: confloat() = -10.0
    temperature_limit_c: confloat(gt=0) = 110.0
    thermal_mass_j_per_c: confloat(gt=0) = 0
    wire_max_safe_current_a: Optional[confloat(gt=0)] = None # Make it optional

    @validator('wire_max_safe_current_a', pre=True, always=True)
    def set_wire_max_safe_current_based_on_csa(v, values):
        # This validator runs if 'wire_max_safe_current_a' is not provided or is None.
        # It requires 'wire' to be already processed by Pydantic.
        if v is None:
            wire_params = values.get('wire')
            if wire_params and isinstance(wire_params, WireParameters):
                csa = wire_params.csa_mm2
                calculated_safe_current = get_wire_max_safe_current(csa)
                if calculated_safe_current == 0.0:
                    # If lookup returns 0, it implies an unknown CSA, which is an issue.
                    raise ValueError(f"Cannot determine max safe current for wire CSA {csa}mm2. Please specify 'wire_max_safe_current_a'.")
                return calculated_safe_current
            # This case should ideally not be hit if 'wire' is a mandatory field in SimulationInput
            # and Pydantic processes fields in order or handles dependencies.
            # If 'wire' isn't available yet, we can't determine the default.
            raise ValueError("Wire parameters not available to determine default max safe current.")
        return v

# Core Calculation Functions
def calculate_current(power_w: float, voltage_v: float) -> float:
    if voltage_v == 0:
        return float('inf') # Avoid division by zero, return infinity
    return power_w / voltage_v

def calculate_wire_resistance(resistivity: float, length_m: float, csa_mm2: float) -> float:
    # Resistance = resistivity * (2 * Length / Area)
    # Length is doubled because it's a two-way path (live and neutral)
    if csa_mm2 == 0:
        return float('inf') # Avoid division by zero
    return resistivity * (2 * length_m) / csa_mm2

def calculate_power_loss(current_a: float, resistance_ohm: float) -> float:
    # Power Loss = I^2 * R
    return (current_a ** 2) * resistance_ohm

def get_wire_max_safe_current(csa_mm2: float) -> float:
    # Simplified lookup table for common copper wire CSA (mm^2) and their approximate max safe current (Amps)
    # These values are illustrative and can vary based on installation method, insulation type, etc.
    # For a real application, consult relevant electrical codes (e.g., IEC, NEC).
    # This function is used as a default if wire_max_safe_current_a is not provided in SimulationInput.
    # It is also used for an internal check if the provided breaker is too large for the wire.
    if math.isclose(csa_mm2, 1.0):
        return 10.0 # Example value
    elif math.isclose(csa_mm2, 1.5):
        return 13.0 # Example value (often cited as 13-16A depending on conditions)
    elif math.isclose(csa_mm2, 2.5):
        return 17.0 # Example value (often cited as 17-23A)
    elif math.isclose(csa_mm2, 4.0):
        return 25.0 # Example value (often cited as 25-30A)
    else:
        # For CSAs not in the table, we can't provide a safe current.
        # This indicates a need for more comprehensive data or user input.
        # Returning 0 or raising an error might be appropriate.
        # For now, let's return a low value to indicate it's not a recognized safe CSA.
        print(f"Warning: No standard max safe current predefined for CSA {csa_mm2}mm2. Please verify independently.")
        return 0.0 # Or consider raising ValueError

# Breaker Tripping Characteristics (based on IEC 60898-1)
# 'thermal_min_trip_current_multiple': multiple of In for conventional non-tripping current (must hold for 1-2 hours)
# 'thermal_max_trip_current_multiple': multiple of In for conventional tripping current (must trip within 1-2 hours)
# 'magnetic_min_trip_current_multiple': minimum multiple of In for instantaneous (magnetic) trip
# 'magnetic_max_trip_current_multiple': maximum multiple of In for instantaneous (magnetic) trip (defines the range)

BREAKER_TRIP_CURVES = {
    "B": {
        "thermal_min_nontrip_current_multiple": 1.13, # Must not trip at this current (conventional time)
        "thermal_max_trip_current_multiple": 1.45,   # Must trip at this current (conventional time, e.g., 1hr for In <= 63A)
        "magnetic_min_trip_current_multiple": 3.0,
        "magnetic_max_trip_current_multiple": 5.0,
    },
    "C": {
        "thermal_min_nontrip_current_multiple": 1.13,
        "thermal_max_trip_current_multiple": 1.45,
        "magnetic_min_trip_current_multiple": 5.0,
        "magnetic_max_trip_current_multiple": 10.0,
    },
    "D": {
        "thermal_min_nontrip_current_multiple": 1.13,
        "thermal_max_trip_current_multiple": 1.45,
        "magnetic_min_trip_current_multiple": 10.0,
        "magnetic_max_trip_current_multiple": 20.0,
    }
    # Note: K and Z types are not included as per the issue's scope.
}

def check_breaker_trip(actual_current_a: float,
                       breaker_rated_current_a: float,
                       breaker_type: str,
                       duration_s: float) -> tuple[bool, str]: # Returns (tripped, reason)
    """
    Checks if a circuit breaker would trip based on current, type, and duration.
    This is a simplified model. Actual trip times vary based on specific I/In ratio.

    Args:
        actual_current_a: The actual current flowing through the circuit.
        breaker_rated_current_a: The nominal rated current of the breaker.
        breaker_type: Type of the breaker ("B", "C", or "D").
        duration_s: The time in seconds the overcurrent condition has persisted.

    Returns:
        A tuple (bool, str) indicating (tripped, reason_for_trip).
        Reason can be "magnetic" or "thermal_overload_time_limit".
    """
    if breaker_rated_current_a <= 0:
        return False, "invalid_breaker_rating" # Should not happen with Pydantic validation

    curve = BREAKER_TRIP_CURVES.get(breaker_type)
    if not curve:
        # This should not happen if breaker_type is validated by Pydantic Enum
        raise ValueError(f"Unknown breaker type: {breaker_type}")

    current_ratio = actual_current_a / breaker_rated_current_a

    # 1. Check for Magnetic Trip (Instantaneous for high overcurrent)
    if current_ratio > curve["magnetic_min_trip_current_multiple"]:
        # For Type B, C, D, this is typically within 0.1s.
        # If current is above the minimum magnetic threshold, assume it trips quickly.
        # A more precise model might check if it's also below magnetic_max_trip_current_multiple,
        # but for safety simulation, if it's above min, it's considered a trip condition.
        return True, "magnetic"

    # 2. Check for Thermal Overload Trip (Simplified time-based)
    # This is a highly simplified representation of the I-t curve.
    # Standard: At 1.45 * In (thermal_max_trip_current_multiple), must trip within 1 hour (for In <= 63A).
    # Standard: At 1.13 * In (thermal_min_nontrip_current_multiple), must NOT trip within 1 hour.

    if current_ratio >= curve["thermal_max_trip_current_multiple"]:
        # Example: If current is 1.45x In, trip time is defined by standard (e.g., <1hr or <2hr).
        # Let's use a simplified threshold: if it persists for a significant duration, it trips.
        # A common reference point: 1.45 * In should trip within 3600s (1 hour).
        # We can scale this down: higher currents trip faster.
        # (1.45 * In / current_ratio) * 3600s might be a rough approximation for time.
        # For example, at 2 * In (if not magnetically tripped): (1.45 / 2) * 3600 = ~2610s

        # Simplified: if current is above the "must trip" thermal level,
        # and it has been on for a duration that's significant.
        # Let's set a placeholder: if current is > 1.45x In and lasts for > 10 minutes (600s), assume trip.
        # This is arbitrary and needs refinement for a real model.
        # The SimPy simulation will call this repeatedly, so 'duration_s' will be the SimPy time elapsed.
        if duration_s > 600: # Arbitrary 10 minutes for sustained high thermal overload
             return True, f"thermal_overload_sustained_at_{current_ratio:.2f}xIn_for_{duration_s}s"

    # More nuanced thermal check:
    # A very rough approximation for time-to-trip t = (k / ( (I/In)^alpha - C ))
    # For simplicity, let's use a few fixed points for this example.
    # If current is 2.5 * In, trip might be around 1-60 seconds for B,C,D.
    if current_ratio > 2.5 and duration_s > 60: # If 2.5x rated current for more than 60s
        return True, f"thermal_overload_high_at_{current_ratio:.2f}xIn_for_{duration_s}s"

    if current_ratio > 1.5 and current_ratio < curve["magnetic_min_trip_current_multiple"]:
        # If between typical thermal and magnetic, e.g., 2x In.
        # Trip time could be minutes. e.g. 120s for 2xIn.
        if duration_s > 120: # If 1.5x-min_magnetic_trip_range for more than 120s
            return True, f"thermal_overload_medium_at_{current_ratio:.2f}xIn_for_{duration_s}s"


    return False, "no_trip"

class KettleSimulation:
    def __init__(self, sim_input: SimulationInput):
        self.sim_input = sim_input
        self.env = simpy.Environment()
        self.results = {
            "status": "Not run",
            "final_wire_temp_c": self.sim_input.ambient_temperature_c,
            "time_to_trip_s": None,
            "time_to_overheat_s": None,
            "total_operation_time_s": 0,
            "trip_reason": None,
            "current_drawn_a": 0,
            "wire_resistance_ohm": 0,
            "power_loss_in_wire_w": 0,
            "initial_assessment": "",
            "thermal_mass_j_per_c": thermal_mass_j_per_c(sim_input.wire.csa_mm2, sim_input.wire.length_m),
        }

        self.results["current_drawn_a"] = calculate_current(
            self.sim_input.load.power_w, self.sim_input.load.voltage_v
        )
        self.results["wire_resistance_ohm"] = calculate_wire_resistance(
            self.sim_input.wire.material_resistivity_ohm_mm2_per_m,
            self.sim_input.wire.length_m,
            self.sim_input.wire.csa_mm2
        )
        if self.results["current_drawn_a"] != float('inf') and self.results["wire_resistance_ohm"] != float('inf'):
            self.results["power_loss_in_wire_w"] = calculate_power_loss(
                self.results["current_drawn_a"], self.results["wire_resistance_ohm"]
            )
        else:
             self.results["power_loss_in_wire_w"] = float('inf')

        print(f"Initializing KettleSimulation. Current: {self.results['current_drawn_a']:.2f}A, Wire R: {self.results['wire_resistance_ohm']:.4f} Ohms, Loss: {self.results['power_loss_in_wire_w']:.2f}W")

    def simulation_process(self):
        wire_temperature_c = self.sim_input.ambient_temperature_c
        time_step_s = 1

        print(f"Starting simulation process. Target duration: {self.sim_input.load.operation_time_s}s. Temp limit: {self.sim_input.temperature_limit_c}°C.")
        print(f"Wire max safe current: {self.sim_input.wire_max_safe_current_a}A. Breaker: {self.sim_input.breaker.current_a}A Type {self.sim_input.breaker.type.value}")

        tripped_at_t0, reason_t0 = check_breaker_trip(
            self.results["current_drawn_a"],
            self.sim_input.breaker.current_a,
            self.sim_input.breaker.type.value,
            0
        )
        if tripped_at_t0 and "magnetic" in reason_t0:
            self.results["status"] = "Breaker Tripped Instantly"
            self.results["time_to_trip_s"] = 0
            self.results["trip_reason"] = reason_t0
            self.results["total_operation_time_s"] = 0
            self.results["final_wire_temp_c"] = wire_temperature_c
            print(f"Event at 0s: Breaker tripped instantly (magnetic). Current: {self.results['current_drawn_a']:.2f}A. Reason: {reason_t0}")
            return

        for target_time in range(time_step_s, int(self.sim_input.load.operation_time_s) + time_step_s, time_step_s):
            yield self.env.timeout(time_step_s)

            current_sim_time = self.env.now
            self.results["total_operation_time_s"] = current_sim_time

            if self.results["thermal_mass_j_per_c"] > 0:
                delta_t = (self.results["power_loss_in_wire_w"] * time_step_s) / self.results["thermal_mass_j_per_c"]
                wire_temperature_c += delta_t
            self.results["final_wire_temp_c"] = wire_temperature_c

            if wire_temperature_c >= self.sim_input.temperature_limit_c:
                self.results["status"] = "Wire Overheated"
                self.results["time_to_overheat_s"] = current_sim_time
                print(f"Event at {current_sim_time}s: Wire overheated. Temp: {wire_temperature_c:.2f}°C")
                return

            tripped, reason = check_breaker_trip(
                self.results["current_drawn_a"],
                self.sim_input.breaker.current_a,
                self.sim_input.breaker.type.value,
                current_sim_time
            )
            if tripped:
                self.results["status"] = "Breaker Tripped"
                self.results["time_to_trip_s"] = current_sim_time
                self.results["trip_reason"] = reason
                print(f"Event at {current_sim_time}s: Breaker tripped. Reason: {reason}. Current: {self.results['current_drawn_a']:.2f}A. Temp: {wire_temperature_c:.2f}°C")
                return

            if current_sim_time > 0 and current_sim_time % 60 == 0 :
                print(f"Sim Time: {current_sim_time}s, Wire Temp: {wire_temperature_c:.2f}°C")

        self.results["status"] = "Completed Normally"
        print(f"Event at {self.results['total_operation_time_s']}s: Kettle operation completed normally. Final Temp: {wire_temperature_c:.2f}°C")

    def run(self):
        print(f"--- Kettle Simulation Run ---")
        print(f"Parameters: {self.sim_input.model_dump_json(indent=2)}")
        print(f"Calculated Current: {self.results['current_drawn_a']:.2f}A")
        print(f"Calculated thermal_mass_j_per_c: {self.results['thermal_mass_j_per_c']:.2f}J/°C")
        print(f"Wire Resistance: {self.results['wire_resistance_ohm']:.4f} Ohms")
        print(f"Power Loss in Wire: {self.results['power_loss_in_wire_w']:.2f}W")

        if self.results["current_drawn_a"] == float('inf') or self.results["wire_resistance_ohm"] == float('inf') or self.results["power_loss_in_wire_w"] == float('inf'):
            self.results["status"] = "Error: Invalid parameters (e.g., V=0 or CSA=0 leading to division by zero)."
            print(f"Critical Error: {self.results['status']}")
        elif self.sim_input.breaker.current_a > self.sim_input.wire_max_safe_current_a:
             self.results["initial_assessment"] = "Unsafe: Breaker rating exceeds wire's maximum safe current."
             self.results["status"] = "Configuration Unsafe (Breaker > Wire)"
             print(f"Configuration Issue: {self.results['initial_assessment']}")
        else:
            self.env.process(self.simulation_process())
            try:
                self.env.run()
            except Exception as e:
                self.results["status"] = f"Simulation Error: {e}"
                print(f"Error during SimPy run: {e}")

        print(f"--- Simulation Finished ---")
        print(f"Final Status: {self.results['status']}")
        if self.results['trip_reason']: print(f"Trip Reason: {self.results['trip_reason']}")
        if self.results['time_to_trip_s'] is not None: print(f"Time to Trip: {self.results['time_to_trip_s']}s")
        if self.results['time_to_overheat_s'] is not None: print(f"Time to Overheat: {self.results['time_to_overheat_s']}s")
        print(f"Total Operation Time: {self.results['total_operation_time_s']}s")
        print(f"Final Wire Temperature: {self.results['final_wire_temp_c']:.2f}°C")

        final_verdict = self.determine_verdict()
        self.results["verdict"] = final_verdict # Store it too
        print(f"--- Final Verdict ---")
        print(f"Verdict: {final_verdict}")
        print(f"-----------------------")

    def determine_verdict(self) -> str:
        verdict = "Undetermined" # Default
        status = self.results.get("status", "Not run")
        current_drawn = self.results.get("current_drawn_a", 0)
        wire_max_safe = self.sim_input.wire_max_safe_current_a
        breaker_rating = self.sim_input.breaker.current_a
        time_to_overheat = self.results.get("time_to_overheat_s")
        time_to_trip = self.results.get("time_to_trip_s")

        # Priority 1: Pre-simulation safety checks from initial_assessment
        if self.results.get("initial_assessment"):
            if "Unsafe" in self.results["initial_assessment"]:
                return f"Unsafe ({self.results['initial_assessment']})"

        # Priority 2: Critical errors during simulation setup
        if "Error:" in status:
            return f"Error in Simulation ({status})"

        # General Unsafe conditions
        if status == "Wire Overheated":
            if current_drawn > wire_max_safe:
                return f"Unsafe (Wire overheated due to current {current_drawn:.2f}A exceeding wire max safe current {wire_max_safe:.2f}A)"
            else:
                # Overheated even if current is technically within wire's rated ampacity.
                # This implies thermal properties (thermal_mass, operation_time) are insufficient for the load.
                return f"Unsafe (Wire overheated at {current_drawn:.2f}A; check thermal properties or operation time)"

        if status == "Completed Normally" and current_drawn > wire_max_safe:
            # Kettle ran for full duration with current exceeding wire's capacity, and breaker didn't trip.
            # This is unsafe because the breaker is not protecting the wire adequately for this load.
            if current_drawn > breaker_rating :
                 # This case implies breaker should have tripped but simulation says it completed normally.
                 # This could point to an issue in breaker trip logic or that the thermal trip time is longer than kettle operation.
                 return f"Unsafe (Current {current_drawn:.2f}A > wire max {wire_max_safe:.2f}A; Breaker {breaker_rating}A did not trip as expected for this overcurrent)"
            else:
                 # Current > wire max, but current <= breaker rating. Breaker is too large for the wire.
                 # This should ideally be caught by `initial_assessment` (breaker_rating > wire_max_safe_current_a)
                 # but this is a double check during runtime.
                 return f"Unsafe (Current {current_drawn:.2f}A > wire max {wire_max_safe:.2f}A, and breaker {breaker_rating}A is too large to protect wire)"


        # At Risk conditions
        if status == "Breaker Tripped":
            if current_drawn > wire_max_safe:
                # Breaker tripped, current was higher than wire's safe limit.
                # Breaker did its job, but the configuration is still risky as it relies on breaker action.
                return f"At Risk (Breaker tripped at {current_drawn:.2f}A, which exceeds wire max safe current {wire_max_safe:.2f}A. Breaker protected wire.)"
            elif time_to_trip is not None and time_to_trip < self.sim_input.load.operation_time_s * 0.5 : # Example: trips very early
                 return f"At Risk (Breaker tripped at {current_drawn:.2f}A. Possible nuisance trip or undersized breaker for load.)"
            else:
                # Breaker tripped, current was within wire's safe limit.
                # This could be a nuisance trip or breaker is close to its rating.
                return f"At Risk (Breaker tripped at {current_drawn:.2f}A. Wire was safe, but operation interrupted. Check breaker sizing for load.)"

        # Safe conditions
        if status == "Completed Normally" and current_drawn <= wire_max_safe:
            # Check if temperature got too close to limit, even if not exceeded.
            final_temp = self.results.get("final_wire_temp_c", self.sim_input.ambient_temperature_c)
            if final_temp > self.sim_input.temperature_limit_c * 0.85: # e.g. > 85% of limit
                return f"Safe (Completed normally, but final wire temperature {final_temp:.2f}°C is high, close to limit {self.sim_input.temperature_limit_c}°C. Consider parameters.)"
            return "Safe (Completed normally, wire temperature and current within limits)"

        if status == "Breaker Tripped Instantly": # Usually means very high current
             if current_drawn > wire_max_safe:
                 return f"At Risk (Breaker tripped instantly at {current_drawn:.2f}A, protecting wire from excessive current)"
             else: # Instant trip but current was "safe" for wire - unusual, points to extreme sensitivity or fault not modeled
                 return f"At Risk (Breaker tripped instantly at {current_drawn:.2f}A. Wire was within ampacity, check for fault or breaker issue.)"


        return f"Undetermined (Simulation status: {status})"

    @staticmethod
    def get_info():
        return "Simulates an electrical circuit with a kettle, wire, and circuit breaker to assess safety."

def run_simulation(params: dict = None):
    if params is None:
        print("Error: No parameters provided for kettle simulation.")
        # Or use default example_params if that's desired for direct runs
        return

    try:
        sim_input_model = SimulationInput(**params)
    except Exception as e: # Catch Pydantic's ValidationError
        print(f"Error: Invalid input parameters: {e}")
        return

    print(f"Successfully parsed input for kettle simulation: {sim_input_model.model_dump_json(indent=2)}")
    sim = KettleSimulation(sim_input=sim_input_model)
    sim.run()

if __name__ == '__main__':
    example_params_dict = {
        "load": {
            "power_w": 2000,
            "voltage_v": 230,
            "operation_time_s": 300
        },
        "wire": {
            "csa_mm2": 1.5, # Should default to 13A max safe current
            "length_m": 10,
            # material_resistivity is optional in model, defaults to 0.0175
        },
        "breaker": {
            "current_a": 13, # Breaker current
            "type": "B"
        }
        # wire_max_safe_current_a is removed to test defaulting logic
    }
    run_simulation(params=example_params_dict)

    # Example of invalid parameters to test validation:
    print("\nTrying with invalid parameters (expecting error):")
    invalid_params_dict = {
        "load": {
            "power_w": -2000, # Invalid
            "voltage_v": 230,
            "operation_time_s": 30 # Invalid, too short
        },
        "wire": {
            "csa_mm2": 0, # Invalid
            "length_m": 100 # Invalid, too long
        },
        "breaker": {
            "current_a": 13,
            "type": "X" # Invalid
        },
        "wire_max_safe_current_a": "abc" # Invalid type
    }
    run_simulation(params=invalid_params_dict)
