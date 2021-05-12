
import numpy as np  
from numpy import linalg as lag
import matplotlib
import matplotlib.animation as animation 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.shape_base import tile
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.spatial.transform import Rotation as R
from IPython.display import HTML
plt.style.use('ggplot')


##USEFUL FUNCTIONS
normal = lambda p: np.array([np.cos(p),np.sin(p)])
normalise = lambda p: p/(np.sqrt(p[0]**2+p[1]**2))           

class Flock():
    def __init__(self,N,speed,frame_size):
        """Frame size should be given in number of radiuses """
        self.frame_size =frame_size
        self.N =N
        self.positions = np.random.uniform(0,self.frame_size,(self.N,2))
        self.thetas = np.random.uniform(0,2*np.pi,self.N)
        self.speed = speed
        self.R =1

    ##GENERAL
    def reset(self):
        """Resets positions and directions to random states """
        self.positions = np.random.uniform(0,self.frame_size,(self.N,2))
        self.thetas = np.random.uniform(0,2*np.pi,self.N)

    def pvec(self):
        """Returns a square matrix of pairwise vectors between all birds """
        #TODO maybe make global function for effecient reuse in pred prey/3d
        p=self.positions
        p_tile_v = np.tile(p,(self.N,1)).reshape(self.N,self.N,2)
        p_tile_h = np.tile(p,(1,self.N)).reshape(self.N,self.N,2)
        return p_tile_v-p_tile_h
    
    ##GETTERS
    def get_birds_in_radius(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius). Works out if x
        or y component differences >L/2 and wraps accordingly. """
        ##TODO find out whether own birds direction should be included
        pvec =self.pvec()
        zeros =np.zeros(pvec.shape)
        L =self.frame_size
        pvec_wrapped =pvec -np.where(np.abs(pvec)>L/2,np.sign(pvec)*L,zeros)
        pdist_wrapped = lag.norm(pvec_wrapped,axis=2)
        indexs =pdist_wrapped<1
        return indexs


    def get_directions(self):
        """Returns array of [[cos(theta),sin(theta)],...] for each theta ie the 
        directions of all the birds"""
        return normal(self.thetas).transpose()


    def update_posdirs(self,dt,sigma,type="vicsek"):
        """Calculates and updates new positions and directions for all the birds """
        if type =="variable":
            speeds =np.tile(self.speed,(2,1)).T
            new_positions = self.positions + self.get_directions()*speeds*dt
        else:
            new_positions = self.positions + self.get_directions()*self.speed*dt
        ##wrapping over frame   
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.get_directions(),(self.N,1,1))
        indexs= self.get_birds_in_radius()
         ##removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0   #TODO could use np.where to be consistent 
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,2),axis=1)
        new_thetas = np.arctan2(direction_sums[:,1],direction_sums[:,0])+np.random.normal(0,sigma,self.N) 

        ##synchronously update positions and thetas
        self.positions = new_positions
        self.thetas = new_thetas
    
    def calculate_order_stat(self):
        """Calculates the vicsek order parameter  """
        d =self.get_directions()
        return lag.norm(np.sum(d,axis=0))/self.N

    ##DISPLAYERS
    def display_state(self,plot_type = "q"):
        """Displays current postions of birds on scatter or quiver plot """
        # with plt.style.context("dark_background"):
        fig,ax = plt.subplots()
        fig.set_size_inches(14, 8)
        p=self.positions
        ax.set(xlim=(0,self.frame_size),ylim = (0,self.frame_size))

        if plot_type =="q":
            d=self.get_directions()
            sf =100
            plot = plt.quiver(p[:,0],p[:,1],d[:,0],d[:,1],scale =sf,color="b")

        elif plot_type =="s":
            plot= plt.scatter(p[:,0],p[:,1],marker =".",color ="b")
            if name:
                plt.savefig(name,dpi = 150)

        else:
            print("Invalid Plot Type. {} is not supported, pls use 'q' or 's'".format(plot_type))
        plt.show()
        return plot
    
    def animate_movement(self,dt,interval,frames,sigma,type="vicsek",args={}):
        """Creates a quiver matplotlib animation of the birds moving """
        ##setting up plot
        fig,ax = plt.subplots()
        fig.set_size_inches(14,8)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))
        init_pos = self.positions
        init_dir =self.get_directions()
        if type=="variable":
            s = self.speed
            color = (0,0,s/np.max(s))
        q =ax.quiver(init_pos[:,0],init_pos[:,1],init_dir[:,0],init_dir[:,1],scale = 100,color="b")

        ##used in funcanimation to update states
        def animate(i):
            if type=="vicsek":
                self.update_posdirs(dt,sigma)
            elif type=="prey":
                predator = args["predator"]
                repulsion_factor = args["repulsion_factor"]
                self.update_posdirs(dt,sigma,predator=predator,repulsion_factor=repulsion_factor)
            elif type == "variable":
                self.update_posdirs(dt,sigma,type="variable")
            p = self.positions
            d = self.get_directions()
            q.set_offsets(p)
            q.set_UVC(d[:,0],d[:,1])

        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames
        )
        return anim
    
    def plot_order_stat(self,dt,T,sigma,display=False,plot=False):
        """Given a time step and time we plot the vicsek order parameter against time """
        num_steps = T//dt

        ##initialising arrays for time and vicsek order parameter
        t = np.linspace(0,T,num_steps)
        vop = np.zeros(num_steps)
        for i in range(num_steps):
            self.update_posdirs(dt,sigma)
            vop[i] = self.calculate_order_stat()

            ##plotting positions of birds 5 times over the course of T
            if display:
                if i%(num_steps//5)==0:
                    print("Flock at time step {}".format(i))
                    self.display_state()

        ##plotting order stat over time
        if plot:
            fig,ax =plt.subplots()
            plot = ax.plot(t,vop)
            plt.show()
        return t,vop
        

class Flock_3d():
    def __init__(self,N,speed,frame_size):
        """Frame size should be given in number of radiuses """
        self.frame_size =frame_size
        self.N =N
        self.positions = np.random.uniform(0,self.frame_size,(self.N,3))

        ##projecting cylinder onto sphere to get even distribution of directions
        thetas = np.random.uniform(0,2*np.pi,self.N)
        zs = np.random.uniform(-1,1,self.N)
        self.directions = np.array([np.sqrt(1-zs**2)*np.cos(thetas),np.sqrt(1-zs**2)*np.sin(thetas),zs]).transpose()

        self.speed = speed
        self.R =1

    ##GETTERS
    def get_birds_in_radius(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius) """
        ##TODO find out whether own birds direction should be included
        ##TODO this is identical to 2d,and pvec only different by d =3. Add dimension parameter to both ?
        pvec =self.pvec()
        zeros =np.zeros(pvec.shape)
        L =self.frame_size
        pvec_wrapped =pvec -np.where(np.abs(pvec)>L/2,np.sign(pvec)*L,zeros)
        pdist_wrapped = lag.norm(pvec_wrapped,axis=2)
        indexs =pdist_wrapped<1
        return indexs
    
    def get_noise(self,sigma):
        theta_noise = np.random.uniform(0,2*np.pi,self.N)
        phi_noise = np.random.normal(0,sigma,self.N)
        ##want to generate noise about symmetric about the x axis, so unconventially have x axis as pole
        ##noise is not strictly normal as elevation angle is biased to the pole, but it is symmetric perpendicular to i^
        noise = np.array([np.cos(phi_noise),np.cos(theta_noise)*np.sin(phi_noise),np.sin(theta_noise)*np.sin(phi_noise)]).transpose()
        # return noise
        return np.zeros(self.directions.shape)

    def get_spherical(self,directions):
        """ """
        x =directions[:,0]
        y =directions[:,1]
        z =directions[:,2]
        x2_y2 =x**2+y**2
        rs = np.sqrt(x2_y2 +z**2)
        #TODO why is this - ?
        phis = -np.arctan2(z,np.sqrt(x2_y2))
        thetas = np.arctan2(y,x)
        return rs,thetas,phis

    def get_rotation_matrices(self,directions):
        """Gets spherical coordinates for directions and then multiplies rotation matrices """
        rs,thetas,phis = self.get_spherical(directions)
        zero = np.zeros(phis.shape)
        one =np.ones(phis.shape)

        ##see analysis for latex rotation matrices used
        ##TODO add latex rotation matrices in analysis
        rot_thetas = np.array([[np.cos(thetas),-np.sin(thetas),zero],[np.sin(thetas),np.cos(thetas),zero],[zero,zero,one]])
        rot_thetas =np.moveaxis(rot_thetas,2,0)
        rot_phis = np.array([[np.cos(phis),zero,np.sin(phis)],[zero,one,zero],[-np.sin(phis),zero,np.cos(phis)]])
        rot_phis =np.moveaxis(rot_phis,2,0)
        return np.matmul(rot_phis,rot_thetas)
    
    def pvec(self):
        """Returns a square matrix of pairwise vectors between all birds """
        ##TODO see GBIR
        p=self.positions
        p_tile_v = np.tile(p,(self.N,1)).reshape(self.N,self.N,3)
        p_tile_h = np.tile(p,(1,self.N)).reshape(self.N,self.N,3)
        return p_tile_v-p_tile_h
    

    def update_posdirs(self,dt,sigma):
        """Calculates and updates new positions and directions for all the birds """
        new_positions = self.positions + self.directions*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.directions,(self.N,1,1))
        indexs= self.get_birds_in_radius()

        ##removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,3),axis=1)
        noise = self.get_noise(sigma)

        ##rotate the noise to the direction_sums and add it 
        rs = self.get_rotation_matrices(direction_sums)
        num_birds = np.sum(indexs,axis=1)
        ##TODO fix this
        # for i in range(self.N):
        #     direction_sums[i] +=num_birds[i]* np.matmul(rs[i],noise)
        

        ##TODO clunky, fix it
        nxd =np.divide(direction_sums[:,0],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        nyd=np.divide(direction_sums[:,1],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        nzd =np.divide(direction_sums[:,2],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        new_directions = np.concatenate((nxd,nyd,nzd),axis=1)
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.directions =new_directions
    
     
    ##DISPLAYERS
    def display_state(self):
        """Creates a 3d matplotlib quiver plot showing positions and directions """
        ##OBSOLETE?
        fig = plt.figure()
        fig.set_size_inches(10,10)
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size),zlim=(0,self.frame_size))
        title = ax.set_title('3D Vicsek Birds')
        p=self.positions
        d = self.directions
        graph =ax.quiver(p[:,0],p[:,1],p[:,2],d[:,0],d[:,1],d[:,2],length=0.6,arrow_length_ratio = 0.8,color="k")
    
    def display_state_plotly(self):
        ##TODO title,labels,get rid of color bar
        p =self.positions
        d=self.directions
        data = np.array([p[:,0],p[:,1],p[:,2],d[:,0],d[:,1],d[:,2]]).T
        df =pd.DataFrame(data,columns=("x","y","z","u","v","w"))
        fig = go.Figure(data = go.Cone(
            x=df['x'],
            y=df['y'],
            z=df['z'],
            u=df['u'],
            v=df['v'],
            w=df['w'],
            colorscale = "blackbody",
            sizemode="absolute",
            sizeref=1))
        fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.8),
                             camera_eye=dict(x=1.2, y=1.2, z=0.6)))
        fig.show()



    def animate_movement(self,sigma,num_frames,ambient_rotation=False):
        """Creates a 3d matplotlib scatter animation of the flock moving over time """
        def update_graph(num):
            self.update_posdirs(1,sigma)
            p = self.positions
            d =self.directions
            graph._offsets3d = (p[:,0], p[:,1], p[:,2])
            title.set_text('3D Vicsek Birds, time={}'.format(num))
            if ambient_rotation:
                ##rotates azimodially starting from -60, the default viewing point
                ax.azim = -60 +num

        fig = plt.figure()
        fig.set_size_inches(10,10)
        ax = fig.add_subplot(111, projection='3d')
        ax.set(xlim=(0,nr),ylim=(0,nr),zlim=(0,nr))
        title = ax.set_title('3D Vicsek Birds')

        p =self.positions
        d = self.directions
        graph = ax.scatter(p[:,0],p[:,1],p[:,2],color="k",marker="^")
        anim = matplotlib.animation.FuncAnimation(fig, update_graph, num_frames, 
                                    interval=40, blit=False)
        return anim

class Prey(Flock):

    def update_posdirs(self,dt,sigma,predator,repulsion_factor =1,verbose=False):
        """Calculates and updates new positions and directions for all the birds,with additional repulsion force from predator """


        ##update via vicsek equations,moving back over the frame if it crosses a boundary
        new_positions = self.positions + self.get_directions()*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.get_directions(),(self.N,1,1))
        indexs= self.get_birds_in_radius()

        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,2),axis=1)

        ##add repulsion factor multiple
        N_pred = predator.shape[0]
        tiled_predator = np.tile(predator,(self.N,1,1))
        tiled_prey =np.tile(self.positions,N_pred).reshape(tiled_predator.shape)         
        repulse = (np.sum(tiled_prey-tiled_predator,axis=1))
        repulse_norm = lag.norm(repulse,axis =1)
        repulse_scaled = N_pred*repulsion_factor*np.divide(repulse,np.tile(repulse_norm**2,(2,1)).transpose())

        direction_sums_repulse = direction_sums+repulse_scaled
        if verbose:
            print(f"Repulse = {repulse}.\n Repulse_norm = {repulse_norm}.\n Repulse_scaled ={repulse_scaled}")

        new_thetas = np.arctan2(direction_sums_repulse[:,1],direction_sums_repulse[:,0])+np.random.normal(0,sigma,self.N) 
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.thetas = new_thetas

class Moth(Flock):
    def update_posdirs(self,dt,sigma,leader,attraction_factor =1,verbose=False):
        """Calculates and updates new positions and directions for all the birds,with additional attraction force from leader. 
        Main use case is pygame """


        ##update via vicsek equations,moving back over the frame if it crosses a boundary
        new_positions = self.positions + self.get_directions()*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.get_directions(),(self.N,1,1))
        indexs= self.get_birds_in_radius()

        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,2),axis=1)

        ##add attraction factor 
        tiled_leader = np.tile(leader,(self.N,1))
        attract = tiled_leader-self.positions
        attract_norm = lag.norm(attract,axis =1)
        attract_scaled = attraction_factor*np.divide(attract,np.tile(attract_norm**2,(2,1)).transpose())
        direction_sums_attract = direction_sums+attract_scaled
        if verbose:
            print(f"attract = {attract}./n attract_norm = {attract_norm}. attract_scaled ={attract_scaled}")

        ##arctan2 gives correct quadrants
        new_thetas = np.arctan2(direction_sums_attract[:,1],direction_sums_attract[:,0])+np.random.normal(0,sigma,self.N)
        self.positions = new_positions
        self.thetas = new_thetas

class Predator(Flock):
    def update_posdirs(self,dt,sigma,prey,attraction_factor =1,verbose=False):
            ##update via vicsek equations,moving back over the frame if it crosses a boundary
        new_positions = self.positions + self.get_directions()*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.get_directions(),(self.N,1,1))
        indexs= self.get_birds_in_radius()

        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,2),axis=1)

        ##add attraction factor for multiple prey
        N_prey = prey.shape[0]
        tiled_prey = np.tile(prey,(self.N,1,1))
        tiled_predator =np.tile(self.positions,N_prey).reshape(tiled_prey.shape)         
        attract = (np.sum(tiled_prey-tiled_predator,axis=1))
        attract_norm = lag.norm(attract,axis =1)
        attract_scaled = N_prey*attraction_factor*np.divide(attract,np.tile(attract_norm**2,(2,1)).transpose())

        direction_sums_attract = direction_sums+attract_scaled
        if verbose:
            print(f"attract = {attract}.\n attract_norm = {attract_norm}.\n attract_scaled ={attract_scaled}")

        new_thetas = np.arctan2(direction_sums_attract[:,1],direction_sums_attract[:,0])+np.random.normal(0,sigma,self.N)   
        self.positions = new_positions
        self.thetas = new_thetas


    def get_prey_in_radius(self,prey):
        """Returns an N_pred by N_prey BOOL matrix, ith row representing which prey are in the ith 
        predators radius"""
        ##just returns prey to check update_posdir
        return prey
    
    def animate_movement_pp(self,prey,frames,sigma,interval,repulsion_factor=1,attraction_factor=1):
        """Creates a quiver matplotlib animation of the birds moving """
        ##setting up plot
        fig,ax = plt.subplots()
        fig.set_size_inches(14,8)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))

        ##get starting states
        init_pos_prey = prey.positions
        init_dir_prey =prey.get_directions()
        init_pos_pred = self.positions
        init_dir_pred =self.get_directions()

        ##plot the initial plot
        qprey =ax.quiver(init_pos_prey[:,0],init_pos_prey[:,1],init_dir_prey[:,0],init_dir_prey[:,1],scale = 100,color="b")
        qpred =ax.quiver(init_pos_pred[:,0],init_pos_pred[:,1],init_dir_pred[:,0],init_dir_pred[:,1],scale = 90,
        color="k",headwidth=5,minshaft=0.9)
        ##used in funcanimation to update states
        def animate(i):
            ##update birds positions 
            prey.update_posdirs(1,sigma,self.positions)
            self.update_posdirs(1,sigma,prey.positions)

            ##update the quiver plots with new positions
            p_prey = prey.positions
            d_prey = prey.get_directions()
            qprey.set_offsets(p_prey)
            qprey.set_UVC(d_prey[:,0],d_prey[:,1])

            p_pred = self.positions
            d_pred = self.get_directions()
            qpred.set_offsets(p_pred)
            qpred.set_UVC(d_pred[:,0],d_pred[:,1])

        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames
        )
        return anim