import tkinter as tk
from tkinter import font, ttk

class SliderController:
    def __init__(self, controller):
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("Parameter Adjustment")
        
        large_font = font.Font(size=16, weight="bold")
        
        num_sliders = 12
        slider_height = 140
        window_height = (num_sliders + 3) * slider_height + 50
        
        screen_width = self.root.winfo_screenwidth()
        window_width = 600
        x_offset = screen_width - window_width - 50
        self.root.geometry(f"{window_width}x{window_height}+{x_offset}+50")

        self.pos_gains_master_slider = self.create_rescalable_slider("Master Pos Gains", 0, 10, 0.1, self.controller.pos_gains[0], self.update_pos_gains_master)
        
        self.pos_gains_sliders = [
            self.create_rescalable_slider(f"Pos Gain {i+1}", 0, 10, 0.1, self.controller.pos_gains[i], self.update_pos_gains)
            for i in range(3)
        ]
        
        self.ori_gains_master_slider = self.create_rescalable_slider("Master Ori Gains", 0, 10, 0.1, self.controller.ori_gains[0], self.update_ori_gains_master)
        
        self.ori_gains_sliders = [
            self.create_rescalable_slider(f"Ori Gain {i+1}", 0, 10, 0.1, self.controller.ori_gains[i], self.update_ori_gains)
            for i in range(3)
        ]
        
        self.damping_ratio_slider = self.create_rescalable_slider("Damping Ratio", 0, 5, 0.1, self.controller.damping_ratio, self.update_damping_ratio)
        self.error_tolerance_pos_slider = self.create_rescalable_slider("Error Tolerance Pos", 0, 0.1, 0.001, self.controller.error_tolerance_pos, self.update_error_tolerance_pos)
        self.error_tolerance_ori_slider = self.create_rescalable_slider("Error Tolerance Ori", 0, 0.1, 0.001, self.controller.error_tolerance_ori, self.update_error_tolerance_ori)
        self.max_pos_error_slider = self.create_rescalable_slider("Max Pos Error", 0, 5, 0.1, self.controller.max_pos_error or 0, self.update_max_pos_error)
        self.max_ori_error_slider = self.create_rescalable_slider("Max Ori Error", 0, 5, 0.1, self.controller.max_ori_error or 0, self.update_max_ori_error)
        self.integration_dt_slider = self.create_rescalable_slider("Integration DT", 0, 1, 0.01, self.controller.integration_dt, self.update_integration_dt)
        
        self.method_var = tk.StringVar(value=self.controller.method)
        method_selector = ttk.Combobox(self.root, textvariable=self.method_var, values=["dynamics", "pinv", "svd", "trans", "dls"], font=large_font)
        method_selector.bind("<<ComboboxSelected>>", self.update_method)
        method_selector.pack(pady=20)

    def create_rescalable_slider(self, label, from_, to, resolution, initial, command):
        slider = tk.Scale(self.root, from_=from_, to=to, resolution=resolution, label=label,
                          orient="horizontal", length=500)
        slider.config(font=font.Font(size=16, weight="bold"))
        slider.set(initial)
        slider.pack(pady=10)
        
        # Attach a listener to detect limit and rescale
        slider.bind("<ButtonRelease-1>", lambda event, s=slider: self.rescale_slider(s, command))
        
        return slider

    def rescale_slider(self, slider, command):
        value = slider.get()
        if value == slider.cget("to"):
            new_to = value * 2
            slider.config(to=new_to)
        elif value == slider.cget("from"):
            new_from = value - (slider.cget("to") - value) * 2
            slider.config(from_=new_from)

        command(value)

    # Update methods for each parameter
    def update_pos_gains_master(self, _):
        master_value = self.pos_gains_master_slider.get()
        for slider in self.pos_gains_sliders:
            slider.set(master_value)
        self.update_pos_gains(None)

    def update_pos_gains(self, _):
        new_pos_gains = tuple(slider.get() for slider in self.pos_gains_sliders)
        self.controller.set_parameters(pos_gains=new_pos_gains)

    def update_ori_gains_master(self, _):
        master_value = self.ori_gains_master_slider.get()
        for slider in self.ori_gains_sliders:
            slider.set(master_value)
        self.update_ori_gains(None)

    def update_ori_gains(self, _):
        new_ori_gains = tuple(slider.get() for slider in self.ori_gains_sliders)
        self.controller.set_parameters(ori_gains=new_ori_gains)

    def update_damping_ratio(self, _):
        new_damping_ratio = self.damping_ratio_slider.get()
        self.controller.set_parameters(damping_ratio=new_damping_ratio)

    def update_error_tolerance_pos(self, _):
        new_error_tolerance_pos = self.error_tolerance_pos_slider.get()
        self.controller.set_parameters(error_tolerance_pos=new_error_tolerance_pos)

    def update_error_tolerance_ori(self, _):
        new_error_tolerance_ori = self.error_tolerance_ori_slider.get()
        self.controller.set_parameters(error_tolerance_ori=new_error_tolerance_ori)

    def update_max_pos_error(self, _):
        new_max_pos_error = self.max_pos_error_slider.get()
        self.controller.set_parameters(max_pos_error=new_max_pos_error)

    def update_max_ori_error(self, _):
        new_max_ori_error = self.max_ori_error_slider.get()
        self.controller.set_parameters(max_ori_error=new_max_ori_error)

    def update_integration_dt(self, _):
        new_integration_dt = self.integration_dt_slider.get()
        self.controller.integration_dt = new_integration_dt

    def update_method(self, event):
        selected_method = self.method_var.get()
        self.controller.set_parameters(method=selected_method)