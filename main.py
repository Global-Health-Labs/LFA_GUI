import os
import tkinter as tk
import tkinter.ttk as ttk
from ttkwidgets import TickScale
from PIL import Image, ImageTk
import pandas as pd
from tkinter import Menu, Label, Toplevel, Entry, filedialog, Button, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter,find_peaks
class popupWindow(object):

    value_global=""
    def __init__(self, master):
        self.top=Toplevel(master)
        self.top.columnconfigure(0, weight=1)
        self.top.columnconfigure(1,weight=3)
        self.value=""

        self.l=Label(self.top,text="Sample Label")
        self.l.grid(column=0,row=0,sticky=tk.W, padx=5, pady=5)

        self.e=Entry(self.top)
        self.e.grid(column=1, row=0, sticky=tk.E, padx=5, pady=5)

        self.b=Button(self.top,text='Ok',command=self.cleanup)
        self.b.grid(column=1, row=3, sticky=tk.E, padx=5, pady=5)

    def cleanup(self):
        self.value=self.e.get()
        popupWindow.value_global=self.value
        self.top.destroy()

class MousePositionTracker(tk.Frame):
    """ Tkinter Canvas mouse position widget. """

    def __init__(self, canvas, root):
        self.canvas = canvas
        self.root = root
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()
        self.original_image = None
        self.count = 0
        self.n = 3
        self.reset()
        self.scale_vals = []
        self.spin_vals = []

        # Options for areas outside rectanglar selection.
        select_outside = dict(dash=(2, 2), fill='red', outline='', state=tk.HIDDEN, stipple='gray25')
        # Separate options for area inside rectanglar selection.
        select_inside = dict(dash=(2, 2), fill='', outline='white', state=tk.HIDDEN)
        # Initial extrema of inner and outer rectangles.
        i_min_x, i_min_y, i_max_x, i_max_y = 0, 0, 1, 1
        o_min_x, o_min_y, o_max_x, o_max_y = 0, 0, self.width, self.height
        self.rects = (
            # Area *outside* selection (inner) rectangle.
            self.canvas.create_rectangle(o_min_x, o_min_y,  o_max_x, i_min_y, **select_outside),
            self.canvas.create_rectangle(o_min_x, i_min_y,  i_min_x, i_max_y, **select_outside),
            self.canvas.create_rectangle(i_max_x, i_min_y,  o_max_x, i_max_y, **select_outside),
            self.canvas.create_rectangle(o_min_x, i_max_y,  o_max_x, o_max_y, **select_outside),
            # Inner rectangle.
            self.canvas.create_rectangle(i_min_x, i_min_y,  i_max_x, i_max_y, **select_inside)
        )


        self.df = pd.DataFrame()

        # Create canvas cross-hair lines.
        xhair_opts = dict(dash=(3, 2), fill='white', state=tk.HIDDEN)
        self.lines = (self.canvas.create_line(0, 0, 0, self.height, **xhair_opts),
                      self.canvas.create_line(0, 0, self.width,  0, **xhair_opts))

    def cur_selection(self):
        return (self.start, self.end)

    def begin(self, event):
        self.hide()
        self.start = (event.x, event.y)  # Remember position (no drawing).

    def update(self, event):
        self.end = (event.x, event.y)
        self._update(event)
        # Current extrema of inner and outer rectangles.
        i_min_x, i_min_y,  i_max_x, i_max_y = self._get_coords()
        o_min_x, o_min_y,  o_max_x, o_max_y = 0, 0,  self.width, self.height
        # Update coords of all rectangles based on these extrema.
        self.canvas.coords(self.rects[0], o_min_x, o_min_y,  o_max_x, i_min_y),
        self.canvas.coords(self.rects[1], o_min_x, i_min_y,  i_min_x, i_max_y),
        self.canvas.coords(self.rects[2], i_max_x, i_min_y,  o_max_x, i_max_y),
        self.canvas.coords(self.rects[3], o_min_x, i_max_y,  o_max_x, o_max_y),
        self.canvas.coords(self.rects[4], i_min_x, i_min_y,  i_max_x, i_max_y),

        for rect in self.rects:  # Make sure all are now visible.
            self.canvas.itemconfigure(rect, state=tk.NORMAL)

    def _update(self, event):
        # Update cross-hair lines.
        self.canvas.coords(self.lines[0], event.x, 0, event.x, self.height)
        self.canvas.coords(self.lines[1], 0, event.y, self.width, event.y)
        self.show()

    def _get_coords(self):
        """ Determine coords of a polygon defined by the start and
            end points one of the diagonals of a rectangular area.
        """
        return (min((self.start[0], self.end[0])), min((self.start[1], self.end[1])),
                max((self.start[0], self.end[0])), max((self.start[1], self.end[1])))

    def reset(self):
        self.start = self.end = None

    def hide(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.HIDDEN)
        self.canvas.itemconfigure(self.lines[1], state=tk.HIDDEN)
        for rect in self.rects:
            self.canvas.itemconfigure(rect, state=tk.HIDDEN)

    def show(self):
        self.canvas.itemconfigure(self.lines[0], state=tk.NORMAL)
        self.canvas.itemconfigure(self.lines[1], state=tk.NORMAL)

    def autodraw(self):
        """Setup automatic drawing; supports command option"""
        self.reset()
        self.canvas.bind("<Button-1>", self.begin)
        self.canvas.bind("<B1-Motion>", self.update)
        self.canvas.bind("<ButtonRelease-1>", self.quit)

    def quit(self, event):
        self.count+=1
        self.sample_label = simpledialog.askstring("Input", "Sample label",
                                        parent=self.root, initialvalue=self.count)
        if self.sample_label:
            self.crop_ROI()
            self.save_coordinates()
            # try:
            #     self.crop_ROI()
            #     self.save_coordinates()
            # except Exception:
            #     messagebox.showerror("Error", "Something is wrong. Please check if a valid image is loaded and/or a valid LFA region is selected")
            #     return
        self.hide()  # Hide cross-hairs.
        self.reset()

    def crop_ROI(self):
        # print(f'start:{self.start}\tend:{self.end}\taspect:{self.canvas.aspect}')
        left, top = [i/self.canvas.aspect for i in self.start]
        right, bottom = [i/self.canvas.aspect for i in self.end]
        # print(f'L:{left}\tR:{right}\tT:{top}\tB:{bottom}')
        # print(f'i_w:{self.original_image.width} x i_h:{self.original_image.height}')
        # try:
        # except AttributeError:
        #     messagebox.showerror("Error","Can't find image. Please open a valid image.")
        #     return
        roi = self.original_image.crop((left, top, right, bottom))

        roi_gray = roi.convert('L')
        nleft, nright = self.calculate_LR_border(roi_gray)
        roi_tight_gray = roi_gray.crop((nleft, 0, nright, roi.size[1]))
        roi_tight_color = roi.crop((nleft, 0, nright, roi.size[1]))
        channel_lines = [255-np.mean(np.asarray(roi_channel),axis=1) for roi_channel in [roi_tight_gray]+list(roi_tight_color.split())]
        line_peaks = [self.find_lfa_peaks(channel_line) for channel_line in channel_lines]

        file_title = self.file_label+'-'+str(self.count)+'-'+self.sample_label
        save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, file_title+'.png'))
        #roi_tight.save(save_path)

        n_channels = len(line_peaks)
        n_peaks = len(line_peaks[0][1])-1
        color_channels = ['gray', 'red', 'green', 'blue']
        features = [f'peak {i}' for i in range(n_peaks)] + ['background']
        data_types = ['index', 'signal']

        fig, axes = plt.subplots(nrows=1, ncols=1+n_channels, sharex=False, sharey=True)

        fig.suptitle(file_title)
        axes[0].imshow(roi_tight_color, aspect='auto')
        axes[0].set_xticks([])
        axes[0].set_ylabel('distance (pixels)')
        # for scale, spin in zip(self.scale_vals, self.spin_vals):
        #     # vals = [(scale - top) * self.canvas.aspect,
        #     #          min(scale - float(spin) - top, top) * self.canvas.aspect,
        #     #          max(scale + float(spin) - top, top) * self.canvas.aspect
        #     #        ]
        #     vals = [scale,
        #              scale - float(spin),
        #              scale + float(spin)
        #            ]
        #     print(f'n:{app.n_lines.get()}')
        #     print(f't:{top}, b:{bottom}, l:{left}, r:{right}')
        #     print(f'ar:{self.canvas.aspect}')
        #     for i, val in enumerate(vals):
        #         if val > 0 or val < len(line_peaks[0][0]):
        #             print(f'i:{i}, val:{val}')
        #             axes[0].axhline(y = val, color = 'r' if i == 0 else 'k', linestyle = '-')
        for i in range(0, n_channels):
            axes[i+1].set_xlabel(f'{color_channels[i]} (signal)')
            axes[i+1].plot(line_peaks[i][0], range(0,len(line_peaks[i][0])), color_channels[i])
            axes[i+1].plot(line_peaks[i][2][:-1], line_peaks[i][1][:-1], 'o', color=color_channels[i])
            axes[i+1].set_xlim([-25,255])
            axes[i+1].grid(True, which='major', color='lightgray')
            axes[i+1].set_xticks([0,100,200])
        fig.subplots_adjust(hspace=0, wspace=0)
        # for ax in axes:
        #      ax.set_aspect(2, share=True)
        fig.savefig(save_path)
        plt.close(fig)

        # n+1 to accommodate background intensity value appended to top 3 peaks
        # peak_labels = [f'peak {i}' for i in range(1,self.n+2)]
        # peak_labels[3] = 'background'
        # df = pd.DataFrame({'sample label': [self.sample_label]*(self.n+1),
        #                    'feature': peak_labels
        #                   })
        # for i, line_peak in enumerate(line_peaks):
        #     df[f'{color_channels[i]} index'] = line_peak[1]
        #     df[f'{color_channels[i]} signal'] = line_peak[2]

        # self.df=pd.concat([self.df, df])
        data=[[' '.join([c,f,t]), line_peaks[i][k+1][j]] for i,c in enumerate(color_channels) for j,f in enumerate(features) for k,t in enumerate(data_types) if f+t!='backgroundindex']
        df = pd.DataFrame([self.count]+[row[1] for row in data], index=['selection']+[row[0] for row in data], columns=[self.sample_label])
        self.df=pd.concat([self.df, df], axis=1)

    def calculate_LR_border(self, image):
        arr=np.asarray(image)
        mean_vertical=np.mean(arr,axis=0)
        gradient=np.gradient(mean_vertical)
        halfpoint=int(gradient.size//2)
        left=np.argmin(gradient[:halfpoint])
        right=np.argmin(gradient[halfpoint:])+halfpoint

        #new_crop_image=image.crop((left,0,right,image.size[1]))
        # new_crop_image.save('3.png')

        return left, right

    def find_lfa_peaks(self, line_profile):
        filtered = savgol_filter(line_profile, 13, 2)
        # switch to returning peaks > 3*sd above background (= 50 lowest values)?
        lowest_length = np.clip(len(filtered)//2, 1, 50)-1
        lowest = np.sort(filtered)[0:lowest_length]
        background = np.mean(lowest) #+ 3*np.std(lowest)
        peaks,_=find_peaks(filtered)
        # peaks,_=find_peaks(filtered, threshold=3*np.std(lowest))
        peak_height=filtered[peaks]
        peak_index_sorted=np.argsort(peak_height)
        peak_loc_sorted=peaks[peak_index_sorted]
        peak_height_sorted=peak_height[peak_index_sorted]

        # # top self.n peaks
        peak_loc_top3=peak_loc_sorted[-self.n:]
        peak_height_top3=peak_height_sorted[-self.n:]

        while len(peak_loc_top3) < 3:
            peak_loc_top3 = np.append(peak_loc_top3, 0)
        while len(peak_height_top3) < 3:
            peak_height_top3 = np.append(peak_height_top3, 0)

        # # sort by peak location
        peak_index_by_location=np.argsort(peak_loc_top3)
        peak_sort_by_location=np.append(peak_loc_top3[peak_index_by_location], 0)
        peak_height_sorted_by_location=np.append(peak_height_top3[peak_index_by_location], background)
        # peak_index_by_location=np.argsort(peak_loc_sorted)
        # peak_sort_by_location=peak_loc_sorted[peak_index_by_location]
        # peak_height_sorted_by_location=peak_loc_sorted[peak_index_by_location]

        return filtered, peak_sort_by_location, peak_height_sorted_by_location

    def save_coordinates(self):
        try:
            self.df.to_csv(self.csv_save_path, index=True)
        except:
            print('')    

    def update_data(self, image, filename):
        self.count = 0
        self.original_image = image
        (self.dir_name, self.file_name) = os.path.split(filename)
        (self.file_label, self.file_ext) = os.path.splitext(self.file_name)
        folder = os.path.normpath(os.path.join(self.dir_name, self.file_label))
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.csv_save_path = os.path.normpath(os.path.join(self.dir_name, self.file_label, self.file_label+'.csv'))
        #self.save_folder=folder

    def resize(self):
        self.width = self.canvas.winfo_width()
        self.height = self.canvas.winfo_height()

    def update_scales(self, scale_vals, spin_vals):
        self.scale_vals, self.spin_vals = scale_vals, spin_vals

class Application(tk.PanedWindow):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        # self.parent = self._nametowidget(self.winfo_parent())
        self.parent = self.master
        # self.create_menu()

        self.controls = ttk.LabelFrame(self, text='Controls', padding=10, width=150)
        self.add(self.controls, padx=10, pady=10)
        self.images = ttk.LabelFrame(self, text='Image')
        self.add(self.images, padx=10, pady=10)

        self.canvas = tk.Canvas(self.images)
        self.canvas.aspect = 1.0
        self.canvas.place(relheight=1.0, relwidth=1.0)
        self.canvas.bind("<Configure>", self.resize_image)

        self.img_path = "./front.png"
        self.img = Image.open(self.img_path)
        self.img_tk = ImageTk.PhotoImage(self.img)
        self.img_container=self.canvas.create_image(0, 0, image=self.img_tk, anchor=tk.NW)

        # Create mouse position tracker that uses the function.
        self.posn_tracker = MousePositionTracker(self.canvas, parent)
        self.posn_tracker.autodraw()
        # self.button=Button(root, text='Save')
        # self.button.pack(expand=True)
        MAX_LINES = 3
        self.n_lines = tk.IntVar()
        self.n_lines.set(MAX_LINES) # set default number of controls on this line
        self.channel_selection = tk.IntVar()
        self.channel_selection.set(1)
        self.open = ttk.Button(self.controls,
                               command=self.open_file,
                               text='Open image file...')
        self.analyze = ttk.Button(self.controls,
                               command=self.auto_analysis,
                               text='Start auto-analysis...')
        self.channel_buttons = [ttk.Radiobutton(self.controls,
                                                # command=self.update_channel_selection,
                                                text=['red', 'green', 'blue', 'gray'][i]+' channel',
                                                variable=self.channel_selection,
                                                value=i+1)
                                for i in range(4)]
        self.line_buttons = [ttk.Radiobutton(self.controls,
                                        command=self.update_line_selection,
                                        text=f'{i+1}-line strip',
                                        variable=self.n_lines,
                                        value=i+1)
                        for i in range(self.n_lines.get())]
        self.entry_vals = [tk.StringVar(self,
                                        'line '+str(i+1))
                           for i in range(MAX_LINES)]
        self.entries = [ttk.Entry(self.controls,
                                  textvariable=self.entry_vals[i])
                        for i in range(MAX_LINES)]
        self.scale_vals = [tk.IntVar(self, 200+25*i) for i in range(MAX_LINES)]
        self.scales = [TickScale(self.controls,
                                 command=lambda value, index=i: self.update_scales(value, index),
                                 variable=self.scale_vals[i],
                                 from_=1,
                                 orient=tk.VERTICAL,
                                 to=self.img.height,
                                 showvalue=True,
                                 resolution=1,
                                 length=100,
                                 labelpos=tk.W)
                       for i in range(MAX_LINES)]
        self.scales_spinboxes = [ttk.Spinbox(self.controls,
                                             command=lambda value=None, index=i: self.update_scales(value, index),
                                            #  format='%d',
                                             from_=1,
                                             to=self.img.height,
                                             textvariable=self.scale_vals[i],
                                             width=5)
                                 for i in range(MAX_LINES)]
        self.spin_vals = [tk.StringVar(self, 15) for _ in range(MAX_LINES)]
        self.spinboxes = [ttk.Spinbox(self.controls,
                                      command=lambda value=None, index=i: self.update_scales(value, index),
                                      from_=1,
                                      to=100,
                                      textvariable=self.spin_vals[i],
                                      width=5)
                          for i in range(MAX_LINES)]
        self.lines = [self.canvas.create_line(0,
                                              scale.get(),
                                              self.canvas.winfo_width(),
                                              scale.get())
                      for scale in self.scales]
        self.rectangles = [self.canvas.create_rectangle(0,
                                                        max(0,
                                                            self.scales[i].get() - float(self.spinboxes[i].get())),
                                                        self.canvas.winfo_width(),
                                                        min(self.canvas.winfo_height(),
                                                            self.scales[i].get() + float(self.spinboxes[i].get())),
                                                        fill='gray',
                                                        stipple='gray25')
                           for i in range(MAX_LINES)]

    def create_controls(self):
        n = 0
        self.open.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        self.analyze.grid(column=0, row=n, columnspan=2, sticky=tk.EW)
        n += 1
        ttk.Separator(self.controls, orient=tk.HORIZONTAL).grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        [button.grid(column=0, row=n+i, columnspan=2, sticky=tk.W) for i, button in enumerate(self.channel_buttons)]
        n += 1 + len(self.channel_buttons)
        ttk.Separator(self.controls, orient=tk.HORIZONTAL).grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        [button.grid(column=0, row=n+i, columnspan=2, sticky=tk.W) for i, button in enumerate(self.line_buttons)]
        n = n + 1 + len(self.line_buttons)
        ttk.Separator(self.controls, orient=tk.HORIZONTAL).grid(row=n, columnspan=2, sticky=tk.EW)
        n += 1
        [entry.grid(column=0, row=n+3*i, columnspan=2, sticky=tk.EW) for i, entry in enumerate(self.entries)]
        n += 1
        [scale_spinbox.grid(column=0, row=n+3*i, sticky=tk.NW) for i, scale_spinbox in enumerate(self.scales_spinboxes)]
        [spinbox.grid(column=1, row=n+3*i, sticky=tk.NE) for i, spinbox in enumerate(self.spinboxes)]
        [scale.grid(column=0, row=n+1+3*i, columnspan=2, sticky=tk.W) for i, scale in enumerate(self.scales)]
            
    def update_scales(self, val, i):
        scale_val = float(self.scales[i].get())
        spin_val = float(self.spinboxes[i].get())
        self.canvas.coords(self.lines[i], 0, int(self.canvas.aspect * scale_val), self.canvas.winfo_width(), int(self.canvas.aspect * scale_val))
        self.canvas.coords(self.rectangles[i],
                            0,
                            max(0, self.canvas.aspect * (scale_val - spin_val)),
                            self.canvas.winfo_width(),
                            min(self.canvas.winfo_height(), self.canvas.aspect * (scale_val + spin_val)))
        self.posn_tracker.update_scales([v.get() for v in self.scales], [v.get() for v in self.spinboxes])
        # print(f'{[v.get() for v in self.scales]}|{scale_val}|{float(self.scales[i].get())}')

    def resize_image(self, event):  # Must accept these arguments.
        c_w, i_w, c_h, i_h = self.canvas.winfo_width(), self.img.width, self.canvas.winfo_height(), self.img.height
        self.canvas.aspect = c_w / i_w if c_w / i_w < c_h / i_h else c_h / i_h
        self.img_tk = ImageTk.PhotoImage(self.img.resize((int(i_w * self.canvas.aspect), int(i_h * self.canvas.aspect)), Image.LANCZOS))
        self.canvas.itemconfig(self.img_container, image=self.img_tk)
        self.posn_tracker.resize()
        [scale.configure(to=self.img.height) for scale in self.scales]
        [scale_spinbox.configure(to=self.img.height) for scale_spinbox in self.scales_spinboxes]

    # def start_drag(self, start, end, **kwarg):  # Must accept these arguments.
    #     self.selection_obj.update(start, end)

    # def end_drag(self, **kwarg):
    #     self.selection_obj.hide()

    # def create_menu(self):
    #     self.menu_bar = Menu(self.parent)
    #     self.file_menu = Menu(self.menu_bar, tearoff=0)
    #     self.file_menu.add_command(
    #         label="Open...", command=self.open_file)
    #     self.menu_bar.add_cascade(label="File", menu=self.file_menu)
    #     self.analysis_menu = Menu(self.menu_bar, tearoff=0)
    #     self.analysis_menu.add_command(
    #         label="Auto-analysis", command=self.auto_analysis)
    #     self.menu_bar.add_cascade(label="Analysis", menu=self.analysis_menu)

    #     self.parent.config(menu=self.menu_bar)

    def update_line_selection(self):
        n = 7 + len(self.line_buttons) + len(self.channel_buttons)
        [entry.grid_forget() for entry in self.entries]
        [scale_spinbox.grid_forget() for scale_spinbox in self.scales_spinboxes]
        [scale.grid_forget() for scale in self.scales]
        [spinbox.grid_forget() for spinbox in self.spinboxes]
        [self.entries[i].grid(column=0, row=n+3*i, columnspan=2) for i in range(self.n_lines.get())]
        n += 1
        [self.scales_spinboxes[i].grid(column=0, row=n+3*i) for i in range(self.n_lines.get())]
        [self.spinboxes[i].grid(column=1, row=n+3*i) for i in range(self.n_lines.get())]
        [self.scales[i].grid(column=0, row=n+1+3*i) for i in range(self.n_lines.get())]

    def open_file(self, event=None):
        self.img_path = filedialog.askopenfilename(defaultextension=".txt",
                                                             filetypes=[("Image files", "*.png"), ("Image files", "*.tif"), ("All Files", "*.*")])
        print(f'{self.img_path}')
        root.title(f'{os.path.basename(self.img_path)}')
        self.img = Image.open(self.img_path)
        self.resize_image(None)
        [self.update_scales(None, i) for i in range(self.n_lines.get())]
        self.posn_tracker.update_data(self.img, self.img_path)

    def auto_analysis(self, event=None):
        if self.img != None:
            y_start = int(290*self.canvas.aspect)
            y_end = int(430*self.canvas.aspect)
            x_start = int(86*self.canvas.aspect)
            x_end = int(self.img.size[0]*self.canvas.aspect)
            spacing = int(87*self.canvas.aspect)
            x_list = [pos for pos in range(x_start, x_end, spacing)]
            if x_list[-1] != x_end:
                x_list = x_list+[x_end]
            for x1, x2 in zip(x_list[:-1], x_list[1:]):
                self.posn_tracker.start = (x1, y_start)
                self.posn_tracker.end = (x2, y_end)
                rectangle = self.canvas.create_rectangle(x1, y_start, x2, y_end)
                self.posn_tracker.quit(None)
                self.canvas.delete(rectangle)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Image Cropper')
    root.state('zoomed')
    root.minsize(1372, 600)

    app = Application(root, orient=tk.HORIZONTAL, sashwidth=5, name="app")
    app.place(anchor=tk.NW, relwidth=1.0, relheight=1.0)
    app.update()
    app.create_controls()

    app.mainloop()
