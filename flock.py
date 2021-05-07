
import numpy as np  
from numpy import linalg as lag
import matplotlib
import matplotlib.animation as animation 
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform,cdist
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
    def reset(self):
        """Resets positions and directions to random states """
        self.positions = np.random.uniform(0,self.frame_size,(self.N,2))
        self.thetas = np.random.uniform(0,2*np.pi,self.N)

    def get_birds_in_radius_old(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius) """
        ##TODO find out whether own birds direction should be included
        ##TODO add periodic boundary conditions
        indexs = squareform(pdist(self.positions))<self.R
        return indexs

    def get_birds_in_radius(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius) """
        ##TODO find out whether own birds direction should be included
        ##TODO add periodic boundary conditions

        #make grid of 4 frames to get periodic conditions
        p=self.positions
        p_fx =p.copy()
        p_fy=p.copy()
        p_fx_fy=p.copy()
        N= self.N
        p_fx[:,0] += np.full(N,self.frame_size)
        p_fy[:,1] += np.full(N,self.frame_size)
        p_fx_fy += np.full(p.shape,self.frame_size)
        grid  = np.concatenate((p,p_fx,p_fy,p_fx_fy))
        grid_indexs = cdist(p,grid)<self.R
        indexs = grid_indexs[:,0:N]+grid_indexs[:,N:2*N]+grid_indexs[:,2*N:3*N]+grid_indexs[:,3*N:4*N]
        return indexs


    def get_directions(self):
        """Returns array of [[cos(theta),sin(theta)],...] for each theta ie the 
        directions of all the birds"""
        return normal(self.thetas).transpose()


    def update_posdirs(self,dt,sigma):
        """Calculates and updates new positions and directions for all the birds """
        new_positions = self.positions + self.get_directions()*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.get_directions(),(self.N,1,1))
        indexs= self.get_birds_in_radius()

        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,2),axis=1)
        new_thetas = np.arctan2(direction_sums[:,1],direction_sums[:,0])+np.random.normal(0,sigma,self.N) 
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.thetas = new_thetas

    def display_state(self,plot_type = "q",name =None):
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
            if name:
                plt.savefig(name,dpi = 150)

        elif plot_type =="s":
            plot= plt.scatter(p[:,0],p[:,1],marker =".",color ="b")
            if name:
                plt.savefig(name,dpi = 150)

        else:
            print("Invalid Plot Type. {} is not supported, pls use 'q' or 's'".format(plot_type))
        plt.show()
        return plot

    def animate_movement(self,dt,interval,frames,sigma,vicsek=False):
        """Creates a matplotlib animation of the birds moving """
        ##setting up plot
        fig,ax = plt.subplots()

        #fig.set_size_inches(18.5, 10.5)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))
        init_pos = self.positions
        scat =ax.scatter(init_pos[0],init_pos[1],marker=".",c="b")

        ##used in funcanimation to update states
        def animate(i):
            if vicsek:
                self.update_posdirs(dt,sigma)
            else:
                self.update_pos(dt)
            pos = self.positions
            scat.set_offsets(pos)
        ##generating animation
        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames
        )
        return anim
    
    def animate_movement_quiver(self,dt,interval,frames,sigma,vicsek=False):
        """Creates a quiver matplotlib animation of the birds moving """
        ##setting up plot
        fig,ax = plt.subplots()
        fig.set_size_inches(14,8)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))
        init_pos = self.positions
        init_dir =self.get_directions()
        q =ax.quiver(init_pos[:,0],init_pos[:,1],init_dir[:,0],init_dir[:,1],scale = 100,color="b")

        ##used in funcanimation to update states
        def animate(i):
            if vicsek:
                self.update_posdirs(dt,sigma)
            else:
                self.update_pos(dt)
            p = self.positions
            d = self.get_directions()
            q.set_offsets(p)
            q.set_UVC(d[:,0],d[:,1])

        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames
        )
        return anim

    def calculate_order_stat(self):
        """Calculates the vicsek order parameter  """
        d =self.get_directions()

        return lag.norm(np.sum(d,axis=0))/self.N
    
    def plot_order_stat(self,dt,T,sigma,display=False,plot=False):
        """Given a time step and time we plot the vicsek order parameter against time """
        num_steps = T//dt
        t = np.linspace(0,T,num_steps)
        vop = np.zeros(num_steps)
        for i in range(num_steps):
            self.update_posdirs(dt,sigma)
            vop[i] = self.calculate_order_stat()

            ##plotting pos of birds
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
        #     """Frame size should be given in number of radiuses """
        self.frame_size =frame_size
        self.N =N
        self.positions = np.random.uniform(0,self.frame_size,(self.N,3))
        #projecting cylinder onto sphere to get even distribution of directions
        thetas = np.random.uniform(0,2*np.pi,self.N)
        zs = np.random.uniform(-1,1,self.N)
        self.directions = np.array([np.sqrt(1-zs**2)*np.cos(thetas),np.sqrt(1-zs**2)*np.sin(thetas),zs]).transpose()
        self.speed = speed
        self.R =1

    def get_birds_in_radius(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius) """
        ##TODO find out whether own birds direction should be included
        ##TODO add periodic boundary conditions
        indexs = squareform(pdist(self.positions))<self.R
        return indexs

    def update_posdirs(self,dt,sigma):
        """Calculates and updates new positions and directions for all the birds """
        new_positions = self.positions + self.directions*self.speed*dt
        new_positions = new_positions %self.frame_size

        tiled_directions = np.tile(self.directions,(self.N,1,1))
        indexs= self.get_birds_in_radius()

        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(self.N,self.N,3),axis=1)
        noise = self.get_noise(sigma)
        ##TODO clunky, fix in morning
        nxd =np.divide(direction_sums[:,0],lag.norm(direction_sums+noise,axis=1)).reshape(self.N,1)
        nyd=np.divide(direction_sums[:,1],lag.norm(direction_sums+noise,axis=1)).reshape(self.N,1)
        nzd =np.divide(direction_sums[:,2],lag.norm(direction_sums+noise,axis=1)).reshape(self.N,1)
        new_directions = np.concatenate((nxd,nyd,nzd),axis=1)
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.directions =new_directions
    
    def add_noise(self,sigma,direction_sums):
        theta_noise = np.random.normal(0,sigma,self.N)
        phi_noise = np.random.normal(0,sigma,self.N)
        phi_noise = phi_noise/np.cos(phi_noise)
        ds_thetas,ds_phis = self.get_theta_phi(direction_sums)

        noise = np.array([np.cos(theta_noise)*np.cos(phi_noise),np.sin(theta_noise)*np.cos(phi_noise),np.sin(phi_noise)]).transpose()
        #returning 0s to test other methods
        return np.zeros(self.directions.shape)
    
    def get_noise(self,sigma):
        theta_noise = np.random.normal(0,sigma,self.N)
        phi_noise = np.random.normal(0,sigma,self.N)
        phi_noise = phi_noise/np.cos(phi_noise)
        noise = np.array([np.cos(theta_noise)*np.cos(phi_noise),np.sin(theta_noise)*np.cos(phi_noise),np.sin(phi_noise)]).transpose()
        # return noise
        ##returning 0s to test other methods
        return np.zeros(self.directions.shape)
    
    def get_theta_phi(self,directions):
        thetas = np.zeros(self.N)
        phis = np.zeros(self.N)
        return thetas,phis
    
    def display_state(self):
        fig = plt.figure()
        fig.set_size_inches(10,10)
        ax = fig.add_subplot(111, projection='3d')
        k=2
        ax.set(xlim=(0,15),ylim=(0,15),zlim=(0,15))
        title = ax.set_title('3D Vicsek Birds')
        p=f_3d.positions
        d = f_3d.directions
        graph =ax.quiver(p[:,0],p[:,1],p[:,2],d[:,0],d[:,1],d[:,2],length=0.6,arrow_length_ratio = 0.8,color="k")