
import numpy as np  
from numpy import linalg as lag
import matplotlib
import matplotlib.animation as animation 
import matplotlib.pyplot as plt
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

    def animate_movement(self,dt,interval,frames,sigma,vicsek=True):
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
    
    def animate_movement_quiver(self,dt,interval,frames,sigma,vicsek=True):
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

        ##rotate the noise to the direction_sums and add it 
        ##super slow, need to fix
        rs = self.get_rotation_matrices(direction_sums)
        for i in range(self.N):
            direction_sums[i] += np.matmul(rs[i],noise[i])
        

        ##TODO clunky, fix in morning
        nxd =np.divide(direction_sums[:,0],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        nyd=np.divide(direction_sums[:,1],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        nzd =np.divide(direction_sums[:,2],lag.norm(direction_sums,axis=1)).reshape(self.N,1)
        new_directions = np.concatenate((nxd,nyd,nzd),axis=1)
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.directions =new_directions
    
    
    def get_noise(self,sigma):
        theta_noise = np.random.normal(0,sigma,self.N)
        phi_noise = np.random.normal(0,sigma,self.N)
        phi_noise = phi_noise/np.cos(phi_noise)
        noise = np.array([np.cos(theta_noise)*np.cos(phi_noise),np.sin(theta_noise)*np.cos(phi_noise),np.sin(phi_noise)]).transpose()
        return noise
        ##returning 0s to test other methods
        # return np.zeros(self.directions.shape)
    
    def get_spherical(self,directions):
        x =directions[:,0]
        y =directions[:,1]
        z =directions[:,2]
        x2_y2 =x**2+y**2
        rs = np.sqrt(x2_y2 +z**2)
        phis = 90-np.arctan2(z,np.sqrt(x2_y2))*360/(2*np.pi)
        thetas = np.arctan2(y,x)*360/(2*np.pi)
        return rs,thetas,phis

    def get_rotation_matrices(self,directions):
        """Gets spherical coordinates for directions and the multiplies rotation matrices """
        rs,thetas,phis = self.get_spherical(directions)
        zero = np.zeros(phis.shape)
        one =np.ones(phis.shape)
        rot_thetas = np.array([[np.cos(thetas),-np.sin(thetas),zero],[np.sin(thetas),np.cos(thetas),zero],[zero,zero,one]])
        rot_thetas =np.moveaxis(rot_thetas,2,0)
        rot_phis = np.array([[np.cos(phis),zero,np.sin(phis)],[zero,one,zero],[-np.sin(phis),zero,np.cos(phis)]])
        rot_phis =np.moveaxis(rot_phis,2,0)
        # print(f"rot_thetas ={rot_thetas},\n rot_phis = {rot_phis}")
        return np.matmul(rot_thetas,rot_phis)
    
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
    
    def animate_movement(self,sigma,num_frames,ambient_rotation=False):
      mpdate_graph(num):
            self.update_posdirs(1,sigma)
            p = self.positions
            d =self.directions
            graph._offsets3d = (p[:,0], p[:,1], p[:,2])
            title.set_text('3D Vicsek Birds, time={}'.format(num))
            #ambient camera rotation
            if ambient_rotation:
                ##updates the azimuth angle
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

        ##add repulsion factor 
        tiled_predator = np.tile(predator,(self.N,1))
        repulse = self.positions-tiled_predator
        repulse_norm = lag.norm(repulse,axis =1)
        repulse_scaled = repulsion_factor*np.divide(repulse,np.tile(repulse_norm**2,(2,1)).transpose())
        direction_sums_repulse = direction_sums+repulse_scaled
        if verbose:
            print(f"Repulse = {repulse}./n Repulse_norm = {repulse_norm}. Repulse_scaled ={repulse_scaled}")

        new_thetas = np.arctan2(direction_sums_repulse[:,1],direction_sums_repulse[:,0])+np.random.normal(0,sigma,self.N) 
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.thetas = new_thetas

class Moth(Flock):
    def update_posdirs(self,dt,sigma,leader,attraction_factor =1,verbose=False):
        """Calculates and updates new positions and directions for all the birds,with additional attraction force from leader """


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

        new_thetas = np.arctan2(direction_sums_attract[:,1],direction_sums_attract[:,0])+np.random.normal(0,sigma,self.N) 
        ##check whether storing more variables in memory reduces                                                                               perfomance      
        self.positions = new_positions
        self.thetas = new_thetas

class Predator(Flock):
    def 