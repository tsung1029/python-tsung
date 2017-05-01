Last login: Tue Sep 29 13:19:12 on ttys001
Franks-iMac:~ franktsung$ history | grep hoff
  160  ssh tsung@hoffman2.idre.ucla.edu -X
  231  ssh tsung@hoffman2.idre.ucla.edu -X
  232  history | grep hoffm
  233  ssh tsung@hoffman2.idre.ucla.edu -X
  271  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  272  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  273  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  275  ssh tsung@hoffman2.idre.ucla.edu -X
  325  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  330  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  336  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  364  history | grep hoff
  365  ssh tsung@hoffman2.idre.ucla.edu -X
  366  ssh tsung@hoffman2.idre.ucla.edu -X
  367  ssh tsung@hoffman2.idre.ucla.edu -X
  369  ssh tsung@hoffman2.idre.ucla.edu -X
  372  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
    #plotme(data)
  382  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  388  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  393  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  403  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  404  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  405  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  406  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  409  ssh hoffman2.idre.ucla.edu -l tsung -X
  411  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  427  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  428  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  430  history | grep hoff
  431  ssh tsung@hoffman2.idre.ucla.edu -X
  434  ssh tsung@hoffman2.idre.ucla.edu -X
  439  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  440  history | grep hoff
  441  ssh tsung@hoffman2.idre.ucla.edu -X
  442  ssh tsung@hoffman2.idre.ucla.edu -X
  443  ssh tsung@hoffman2.idre.ucla.edu -X
  444  ssh tsung@hoffman2.idre.ucla.edu -X
  445  ssh tsung@hoffman2.idre.ucla.edu -X
  446  ssh tsung@hoffman2.idre.ucla.edu -X
  447  ssh tsung@hoffman2.idre.ucla.edu -X
  448  ssh tsung@hoffman2.idre.ucla.edu -X
  449  ssh tsung@hoffman2.idre.ucla.edu -X
  451  ssh tsung@hoffman2.idre.ucla.edu -X
  452  ssh tsung@hoffman2.idre.ucla.edu -X
  453  ssh tsung@hoffman2.idre.ucla.edu -X
  454  ssh tsung@hoffman2.idre.ucla.edu -X
  456  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  458  ssh tsung@hoffman2.idre.ucla.edu -X
  460  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  461  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  472  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  473  history | grep hoff
  474  ssh hoffman2.idre.ucla.edu -l tsung -X
  481  history | grep hoffm
  482  ssh tsung@hoffman2.idre.ucla.edu -X
  483  ssh tsung@hoffman2.idre.ucla.edu -X
  484  ssh tsung@hoffman2.idre.ucla.edu -X
  485  ssh tsung@hoffman2.idre.ucla.edu -X
  486  ssh tsung@hoffman2.idre.ucla.edu -X
  498  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  499  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt2
  500  sshfs tsung@hoffman2.idre.ucla.edu:/u/d2scratch/tsung mnt
  501  history | grep hoff
Franks-iMac:~ franktsung$ !160
ssh tsung@hoffman2.idre.ucla.edu -X
tsung@hoffman2.idre.ucla.edu's password:
Warning: untrusted X11 forwarding setup failed: xauth key data not generated
Last login: Tue Sep 29 12:42:10 2015 from tp11.physics.ucla.edu
Welcome to the Hoffman2 Cluster!

Hoffman2 Home Page:     http://www.hoffman2.idre.ucla.edu
Consulting:             https://support.idre.ucla.edu/helpdesk

All login nodes should be accessed via "hoffman2.idre.ucla.edu".

Please do NOT compute on the login nodes.

Processes running on the login nodes which seriously degrade others'
use of the system may be terminated without warning.  Use qrsh to obtain
an interactive shell on a compute node for CPU or I/O intensive tasks.

The following news items are currently posted:

   IDRE Fall 2015 Classes
   Purchased storage migration of /u/home/groupname directories
   News Archive On Web Site

Enter shownews to read the full text of a news item.
-bash-4.1$ vi test.py
-bash-4.1$
-bash-4.1$
-bash-4.1$ !pyt
python test.py
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from hdf_adam import *
  File "/u/home/t/tsung/hdf_adam.py", line 7, in <module>
    from pylab import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/pylab.py", line 1, in <module>
    from matplotlib.pylab import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pylab.py", line 265, in <module>
    from matplotlib.pyplot import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pyplot.py", line 97, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/__init__.py", line 25, in pylab_setup
    globals(),locals(),[backend_name])
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtkagg.py", line 10, in <module>
    from matplotlib.backends.backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py", line 13, in <module>
    import gtk; gdk = gtk.gdk
  File "/usr/lib64/python2.6/site-packages/gtk-2.0/gtk/__init__.py", line 64, in <module>
    _init()
  File "/usr/lib64/python2.6/site-packages/gtk-2.0/gtk/__init__.py", line 52, in _init
    _gtk.init_check()
RuntimeError: could not open display
-bash-4.1$ exit
logout


#        tight_layout()
    side_1=plt.subplot(gs[0,3])
    filename_in = my_path+'/MS/PHA/p1x1/species_1/p1x1-species_1-'+strnum+'.h5'
    # print "Reading " + filename_in
    data2 = read_hdf(filename_in)
    # temp=numpy.log10(numpy.absolute(data2.data)+0.00001)
    # data2.data=temp
 #   numpy.fliplr(data2.data)
 #   dx1_2=data2.axes[0].increment
 #   dx2_2=data2.axes[1].increment
    extent_stuff = [    data2.axes[0].axis_min, data2.axes[0].axis_max, data2.axes[1].axis_min, data2.axes[1].axis_max]
    data2.data=numpy.abs(data2.data)
    plot_object2 = side_1.imshow(data2.data, extent=extent_stuff ,aspect='auto',  origin='lower',interpolation='bilinear', cmap='Accent',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=300))
    cb2=colorbar(plot_object2)
    #cb2.set_label("%s \n %s"% ( data2.data_attributes['LONG_NAME'], data2.data_attributes['UNITS']) ,fontsize=mylabelsize)
    #cb2.set_label("p1x1 [a.u.]" ,fontsize=mylabelsize)
    cb2.set_label("p1x1 [a.u.]" ,font=my_font)
    #plt.xlabel("$%s$   $%s$" % (data2.axes[0].attributes['LONG_NAME'][0], math_string(data2.axes[0].attributes['UNITS'])[0] ),fontsize=myfontsize)
    #plt.ylabel("$%s$   $%s$" % (data2.axes[1].attributes['LONG_NAME'][0], math_string(data2.axes[1].attributes['UNITS'][0] )),fontsize=myfontsize)
    plt.ylabel('$p_1 [m_e c]$',fontsize=myfontsize)
    plt.xlabel('$x_1 [c/\omega_0]$',fontsize=myfontsize)
    side_1.set_xticks([0, 250, 500])
#    plot.tight_layout()


    #        tight_layout()
    side_2=plt.subplot(gs[1,3])
    filename_in = my_path+'/MS/PHA/p1p2/species_1/p1p2-species_1-'+strnum+'.h5'
    # print "Reading " + filename_in
    data3 = read_hdf(filename_in)
    # temp=numpy.log10(numpy.absolute(data2.data)+0.00001)
    # data2.data=temp
    # numpy.fliplr(data3.data)
 #   dx1_2=data2.axes[0].increment
 #   dx2_2=data2.axes[1].increment
    extent_stuff = [    data3.axes[0].axis_min, data3.axes[0].axis_max, data3.axes[1].axis_min, data3.axes[1].axis_max]
    data3.data=numpy.abs(data3.data)
    plot_object3 = side_2.imshow(data3.data, extent=extent_stuff ,aspect='auto',  origin='lower',interpolation='bilinear', cmap='Accent',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=300))
    cb3=colorbar(plot_object3)
    #cb2.set_label("%s \n %s"% ( data2.data_attributes['LONG_NAME'], data2.data_attributes['UNITS']) ,fontsize=mylabelsize)
    #cb3.set_label("Electron p1p2 [a.u.]" ,fontsize=mylabelsize)
    cb3.set_label("Electron p1p2 [a.u.]" ,font=my_font)
    #plt.xlabel("$%s$   $%s$" % (data2.axes[0].attributes['LONG_NAME'][0], math_string(data2.axes[0].attributes['UNITS'])[0] ),fontsize=myfontsize)
    #plt.ylabel("$%s$   $%s$" % (data2.axes[1].attributes['LONG_NAME'][0], math_string(data2.axes[1].attributes['UNITS'][0] )),fontsize=myfontsize)
    plt.ylabel('$p_1 [m_e c]$',fontsize=myfontsize)
    plt.xlabel('$p_2 [m_e c]$',fontsize=myfontsize)
#    plot.tight_layout()
    side_2.set_xticks([-0.5,0.0,0.5])

    plt.tight_layout()
    show()
    filename_out=my_path+'/MS/FLD/e1/movie-'+strnum+'.png'

    #plt.savefig(filename_out,dpi=300,bbox_inches='tight')
    matplotlib.pyplot.close("all")









"test.py" 163L, 6970C                                                                                     153,5         Bot
Connection to hoffman2.idre.ucla.edu closed.
Franks-iMac:~ franktsung$ !ss
ssh tsung@hoffman2.idre.ucla.edu -X
tsung@hoffman2.idre.ucla.edu's password:
Warning: untrusted X11 forwarding setup failed: xauth key data not generated
Last login: Tue Sep 29 22:01:22 2015 from tp11.physics.ucla.edu
Welcome to the Hoffman2 Cluster!

Hoffman2 Home Page:     http://www.hoffman2.idre.ucla.edu
Consulting:             https://support.idre.ucla.edu/helpdesk

All login nodes should be accessed via "hoffman2.idre.ucla.edu".

Please do NOT compute on the login nodes.

#
Processes running on the login nodes which seriously degrade others'
use of the system may be terminated without warning.  Use qrsh to obtain
an interactive shell on a compute node for CPU or I/O intensive tasks.

The following news items are currently posted:

   IDRE Fall 2015 Classes
   Purchased storage migration of /u/home/groupname directories
   News Archive On Web Site

Enter shownews to read the full text of a news item.
-bash-4.1$ vi test.py
-bash-4.1$ !pyt
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[35651,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login3

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Traceback (most recent call last):
  File "test.py", line 92, in <module>
    cb.set_label("$E_1 [{m_e c \omega_0}/{q_e}]$" ,font=my_font)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 424, in set_label
    self._set_label()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 414, in _set_label
    self.ax.set_ylabel(self._label, **self._labelkw)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axes.py", line 3341, in set_ylabel
    return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axis.py", line 1387, in set_label_text
    self.label.update(kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/artist.py", line 670, in update
    raise AttributeError('Unknown property %s' % k)
AttributeError: Unknown property font
-bash-4.1$ jobs
-bash-4.1$ vi test.py
-bash-4.1$ !pyt
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[44490,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login3

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Traceback (most recent call last):
  File "test.py", line 91, in <module>
    cb.set_label("$E_1 [{m_e c \omega_0}/{q_e}]$" ,font=my_font)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 424, in set_label
    self._set_label()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 414, in _set_label
    self.ax.set_ylabel(self._label, **self._labelkw)
from hdf_adam import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axes.py", line 3341, in set_ylabel
    return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axis.py", line 1387, in set_label_text
    self.label.update(kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/artist.py", line 670, in update
    raise AttributeError('Unknown property %s' % k)
AttributeError: Unknown property font
-bash-4.1$ !vi
vi test.py
-bash-4.1$ !pyth
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[44820,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login3

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Traceback (most recent call last):
  File "test.py", line 116, in <module>
    cb2.set_label("p1x1 [a.u.]" ,font=my_font)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 424, in set_label
    self._set_label()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/colorbar.py", line 414, in _set_label
    self.ax.set_ylabel(self._label, **self._labelkw)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axes.py", line 3341, in set_ylabel
    return self.yaxis.set_label_text(ylabel, fontdict, **kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axis.py", line 1387, in set_label_text
    self.label.update(kwargs)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/artist.py", line 670, in update
    raise AttributeError('Unknown property %s' % k)
AttributeError: Unknown property font
    #main = plt.subplot2grid((2,2),(0,0),rowspan=2)
    main=plt.subplot(gs[:,0:2])
    plot_object = main.imshow(data.data, extent=extent_stuff ,aspect='auto', origin='lower',interpolation='bilinear',
                              cmap='Rainbow',clim=[-3,4.5])
    xlim(0,1.5)
    ylim(0,0.3)
    #plot_object = imshow(data.data, extent=extent_stuff, aspect='auto', cmap='Rainbow',clim=[-0.05,0.05])
    cb=colorbar(plot_object)
#    cb.set_label("%s \n %s"% ( data.data_attributes['LONG_NAME'], data.data_attributes['UNITS']) ,fontsize=mylabelsize)
    cb.set_label("$E_1 [{m_e c \omega_0}/{q_e}]$" ,fontdict=my_font)
    #cb.set_label("$E_1 [{m_e c \omega_0}/{q_e}]$" ,fontdict=my_font)
    main.set_xticks([0,0.5,1],fontdict=my_font)

    plt.xlabel("$%s$   $%s$" % (data.axes[0].attributes['LONG_NAME'][0], math_string(data.axes[0].attributes['UNITS'])[0] ),fontdict=my_label_font)
    plt.ylabel("$%s$   $%s$" % (data.axes[1].attributes['LONG_NAME'][0], math_string(data.axes[1].attributes['UNITS'][0] )),fontdict=my_label_font)
    #ylabel("%s \n %s"% ( data.data_attributes['LONG_NAME'], data.data_attributes['UNITS']) )


#        tight_layout()
    side_1=plt.subplot(gs[0,3])
    filename_in = my_path+'/MS/PHA/p1x1/species_1/p1x1-species_1-'+strnum+'.h5'
    # print "Reading " + filename_in
    data2 = read_hdf(filename_in)
    # temp=numpy.log10(numpy.absolute(data2.data)+0.00001)
    # data2.data=temp
 #   numpy.fliplr(data2.data)
 #   dx1_2=data2.axes[0].increment
 #   dx2_2=data2.axes[1].increment
    extent_stuff = [    data2.axes[0].axis_min, data2.axes[0].axis_max, data2.axes[1].axis_min, data2.axes[1].axis_max]
    data2.data=numpy.abs(data2.data)
    plot_object2 = side_1.imshow(data2.data, extent=extent_stuff ,aspect='auto',  origin='lower',interpolation='bilinear', cmap='Accent',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=300))
    cb2=colorbar(plot_object2)
    #cb2.set_label("%s \n %s"% ( data2.data_attributes['LONG_NAME'], data2.data_attributes['UNITS']) ,fontsize=mylabelsize)
    #cb2.set_label("p1x1 [a.u.]" ,fontsize=mylabelsize)
    cb2.set_label("p1x1 [a.u.]" ,fontdict=my_font)
    #plt.xlabel("$%s$   $%s$" % (data2.axes[0].attributes['LONG_NAME'][0], math_string(data2.axes[0].attributes['UNITS'])[0] ),fontsize=myfontsize)
    #plt.ylabel("$%s$   $%s$" % (data2.axes[1].attributes['LONG_NAME'][0], math_string(data2.axes[1].attributes['UNITS'][0] )),fontsize=myfontsize)
    plt.ylabel('$p_1 [m_e c]$',fontdict = my_label_font)
    plt.xlabel('$x_1 [c/\omega_0]$',fontdict = my_label_font)
    side_1.set_xticks([0, 250, 500])
#    plot.tight_layout()


    #        tight_layout()
    side_2=plt.subplot(gs[1,3])
    filename_in = my_path+'/MS/PHA/p1p2/species_1/p1p2-species_1-'+strnum+'.h5'
    # print "Reading " + filename_in
    data3 = read_hdf(filename_in)
    # temp=numpy.log10(numpy.absolute(data2.data)+0.00001)
    # data2.data=temp
    # numpy.fliplr(data3.data)
 #   dx1_2=data2.axes[0].increment
 #   dx2_2=data2.axes[1].increment
    extent_stuff = [    data3.axes[0].axis_min, data3.axes[0].axis_max, data3.axes[1].axis_min, data3.axes[1].axis_max]
    data3.data=numpy.abs(data3.data)
    plot_object3 = side_2.imshow(data3.data, extent=extent_stuff ,aspect='auto',  origin='lower',interpolation='bilinear', cmap='Accent',norm=matplotlib.colors.LogNorm(vmin=1e-4,vmax=300))
    cb3=colorbar(plot_object3)
    #cb2.set_label("%s \n %s"% ( data2.data_attributes['LONG_NAME'], data2.data_attributes['UNITS']) ,fontsize=mylabelsize)
    #cb3.set_label("Electron p1p2 [a.u.]" ,fontsize=mylabelsize)
    cb3.set_label("Electron p1p2 [a.u.]" ,fontdict=my_font)
    #plt.xlabel("$%s$   $%s$" % (data2.axes[0].attributes['LONG_NAME'][0], math_string(data2.axes[0].attributes['UNITS'])[0] ),fontsize=myfontsize)
    #plt.ylabel("$%s$   $%s$" % (data2.axes[1].attributes['LONG_NAME'][0], math_string(data2.axes[1].attributes['UNITS'][0] )),fontsize=myfontsize)
    plt.ylabel('$p_1 [m_e c]$',fontdict = my_label_font)
    plt.xlabel('$p_2 [m_e c]$',fontdict = my_label_font)
search hit BOTTOM, continuing at TOP

-bash-4.1$ job
-bash: job: command not found
-bash-4.1$ jobs
-bash-4.1$
-bash-4.1$ !vi
vi test.py
-bash-4.1$ !pyth
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[43355,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login3

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015030.h5
^CTraceback (most recent call last):
  File "test.py", line 150, in <module>
    show()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pyplot.py", line 143, in show
    _show(*args, **kw)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backend_bases.py", line 108, in __call__
    self.mainloop()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py", line 83, in mainloop
    gtk.main()
KeyboardInterrupt
-bash-4.1$ ^C
-bash-4.1$ ^C
-bash-4.1$ !vi
vi test.py
-bash-4.1$ !pyth
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[54640,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login3

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015030.h5
/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py:83: GtkWarning: Attempting to store changes into `/u/home/t/tsung/.local/share/recently-used.xbel', but failed: Failed to create file '/u/home/t/tsung/.local/share/recently-used.xbel.XNJD5X': No such file or directory
  gtk.main()
/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py:83: GtkWarning: Attempting to set the permissions of `/u/home/t/tsung/.local/share/recently-used.xbel', but failed: No such file or directory
  gtk.main()
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015060.h5
^CTraceback (most recent call last):
  File "test.py", line 150, in <module>
    show()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pyplot.py", line 143, in show
    _show(*args, **kw)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backend_bases.py", line 108, in __call__
    self.mainloop()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py", line 83, in mainloop
    gtk.main()
KeyboardInterrupt
-bash-4.1$ ^C
-bash-4.1$ ^C
-bash-4.1$ jobs
-bash-4.1$ vi test.py
-bash-4.1$ !pyt
python test.py
Traceback (most recent call last):
  File "test.py", line 1, in <module>
    from hdf_adam import *
  File "/u/home/t/tsung/hdf_adam.py", line 7, in <module>
    from pylab import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/pylab.py", line 1, in <module>
    from matplotlib.pylab import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pylab.py", line 265, in <module>
    from matplotlib.pyplot import *
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pyplot.py", line 97, in <module>
    _backend_mod, new_figure_manager, draw_if_interactive, _show = pylab_setup()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/__init__.py", line 25, in pylab_setup
    globals(),locals(),[backend_name])
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtkagg.py", line 10, in <module>
    from matplotlib.backends.backend_gtk import gtk, FigureManagerGTK, FigureCanvasGTK,\
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_gtk.py", line 13, in <module>
    import gtk; gdk = gtk.gdk
  File "/usr/lib64/python2.6/site-packages/gtk-2.0/gtk/__init__.py", line 64, in <module>
    _init()
  File "/usr/lib64/python2.6/site-packages/gtk-2.0/gtk/__init__.py", line 52, in _init
    _gtk.init_check()
RuntimeError: could not open display
-bash-4.1$ exit
logout
#
Connection to hoffman2.idre.ucla.edu closed.
Franks-iMac:~ franktsung$ !ss
ssh tsung@hoffman2.idre.ucla.edu -X
tsung@hoffman2.idre.ucla.edu's password:
Warning: untrusted X11 forwarding setup failed: xauth key data not generated
Last login: Mon Sep 28 23:48:06 2015 from tp11.physics.ucla.edu
Welcome to the Hoffman2 Cluster!

Hoffman2 Home Page:     http://www.hoffman2.idre.ucla.edu
Consulting:             https://support.idre.ucla.edu/helpdesk

All login nodes should be accessed via "hoffman2.idre.ucla.edu".

Please do NOT compute on the login nodes.

Processes running on the login nodes which seriously degrade others'
use of the system may be terminated without warning.  Use qrsh to obtain
an interactive shell on a compute node for CPU or I/O intensive tasks.

The following news items are currently posted:

   IDRE Fall 2015 Classes
   Purchased storage migration of /u/home/groupname directories
   News Archive On Web Site

Enter shownews to read the full text of a news item.
-bash-4.1$ !pyth
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[25709,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

Module: OpenFabrics (openib)
  Host: login4

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
Traceback (most recent call last):
  File "test.py", line 99, in <module>
    main.set_xticks([0,0.5,1],fontdict=my_font)
TypeError: set_xticks() got an unexpected keyword argument 'fontdict'
-bash-4.1$ jobs
-bash-4.1$ !vi
vi test.py
-bash-4.1$ !pyt
python test.py
librdmacm: Fatal: no RDMA devices found
--------------------------------------------------------------------------
[[26067,1],0]: A high-performance Open MPI point-to-point messaging module
was unable to find any relevant network interfaces:

 30 # The font.weight property has effectively 13 values: normal, bold,
Module: OpenFabrics (openib)
  Host: login4

Another transport will be used instead, although this may result in
lower performance.
--------------------------------------------------------------------------
Reading /u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed//MS/FLD/e1/e1-015000.h5
$c / \omega_p$
$c$
Traceback (most recent call last):
  File "test.py", line 155, in <module>
    plt.tight_layout()
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/pyplot.py", line 1143, in tight_layout
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/figure.py", line 1536, in tight_layout
    rect=rect)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/tight_layout.py", line 352, in get_tight_layout_figure
    pad=pad, h_pad=h_pad, w_pad=w_pad)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/tight_layout.py", line 129, in auto_adjust_subplotpars
    tight_bbox_raw = union([ax.get_tightbbox(renderer) for ax in subplots])
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axes.py", line 8846, in get_tightbbox
    bb_yaxis = self.yaxis.get_tightbbox(renderer)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/axis.py", line 1027, in get_tightbbox
    bb.append(a.get_window_extent(renderer))
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/text.py", line 752, in get_window_extent
    bbox, info = self._get_layout(self._renderer)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/text.py", line 313, in _get_layout
    ismath=ismath)
  File "/u/local/apps/python/2.6/lib64/python2.6/site-packages/matplotlib/backends/backend_agg.py", line 204, in get_text_wifrom hdf_adam import *

import numpy

# gridspec for making sub-plots
import matplotlib.gridspec as gridspec

#mpi4py for the parallel part
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



myfontsize=8
mylabelsize=8
# The font.family property has five values: 'serif' (e.g., Times),
# 'sans-serif' (e.g., Helvetica), 'cursive' (e.g., Zapf-Chancery),
# 'fantasy' (e.g., Western), and 'monospace' (e.g., Courier).  Each of
# these font families has a default list of font names in decreasing
# order of priority associated with them.  When text.usetex is False,
# font.family may also be one or more concrete font names.
#
# The font.style property has three values: normal (or roman), italic
# or oblique.  The oblique style will be used for italic, if it is not
# present.
#
# The font.variant property has two values: normal or small-caps.  For
# TrueType fonts, which are scalable fonts, small-caps is equivalent
# to using a font size of 'smaller', or about 83% of the current font
# size.
#
# The font.weight property has effectively 13 values: normal, bold,
# bolder, lighter, 100, 200, 300, ..., 900.  Normal is the same as
# 400, and bold is 700.  bolder and lighter are relative values with
# respect to the current weight.
#
# The font.stretch property has 11 values: ultra-condensed,
# extra-condensed, condensed, semi-condensed, normal, semi-expanded,
# expanded, extra-expanded, ultra-expanded, wider, and narrower.  This
# property is not currently implemented.

my_font = {     'family' : 'sans-serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 10
}

my_label_font = {       'family' : 'sans-serif',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 7
}



ibegin=15000
iend=18000
i_int=30

starting = ibegin+rank*i_int

#plt.style.use('fivethirtyeight')
#plt.style.use('bmh')
#plt.style.use('ggplot')
my_path = '/u/d2scratch/tsung/hfhi-paper/5kev-c1-fixed/'
for filenumber in range(starting,iend,i_int*size):

    strnum = str(filenumber)
    strlen = len(strnum)
                                                                                                          1,1           Top