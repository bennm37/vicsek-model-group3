from flock import *
class Prey(Flock):
    def __init__(self, N, speed, frame_size):
        super().__init__(N, speed, frame_size)
        self.alive = np.full(self.N,True)
    def pvec_alive(self):
        """Returns a square matrix of pairwise vectors between all alive birds """
        N_alive = np.sum(self.alive)
        p=self.positions[self.alive]
        p_tile_v = np.tile(p,(N_alive,1)).reshape(N_alive,N_alive,2)
        p_tile_h = np.tile(p,(1,N_alive)).reshape(N_alive,N_alive,2)
        return p_tile_v-p_tile_h

    def get_alive_birds_in_radius(self):
        """Returns NxN BOOL array of which birds are in radius (ie the Nth row 
        is an array of which birds are in the Nth birds radius). Works out if x
        or y component differences >L/2 and wraps accordingly. """
        pvec =self.pvec_alive()
        zeros =np.zeros(pvec.shape)
        L =self.frame_size
        pvec_wrapped =pvec -np.where(np.abs(pvec)>L/2,np.sign(pvec)*L,zeros)
        pdist_wrapped = lag.norm(pvec_wrapped,axis=2)
        indexs =pdist_wrapped<1
        return indexs

    def kill(self,dead):
        """Replaces dead birds positions with None """
        self.positions[dead]=[None,None]
        self.alive[dead] = False

    def update_posdirs(self,dt,sigma,predator,repulsion_factor =1,verbose=False):
        """Calculates and updates new positions and directions for all the birds,with additional repulsion force from predator """
        N_alive = np.sum(self.alive)
        p_alive = self.positions[self.alive]
        d_alive = self.get_directions()[self.alive]
        ##update via vicsek equations,moving back over the frame if it crosses a boundary
        new_positions = p_alive + d_alive*self.speed*dt
        new_positions = new_positions %self.frame_size



        tiled_directions = np.tile(d_alive,(N_alive,1,1))
        indexs= self.get_alive_birds_in_radius()
        if verbose:
            print(tiled_directions.shape)
            print(indexs.shape)
            print(f"N alive = {N_alive}")
        #removes all birds not in radius from the sum
        tiled_directions[np.invert(indexs)] *= 0  
        direction_sums = np.sum(tiled_directions.reshape(N_alive,N_alive,2),axis=1)

        ##add repulsion factor multiple
        N_pred = predator.shape[0]
        tiled_predator = np.tile(predator,(N_alive,1,1))
        tiled_prey =np.tile(p_alive,N_pred).reshape(tiled_predator.shape)         
        repulse = (np.sum(tiled_prey-tiled_predator,axis=1))
        repulse_norm = lag.norm(repulse,axis =1)
        repulse_scaled = N_pred*repulsion_factor*np.divide(repulse,np.tile(repulse_norm**2,(2,1)).transpose())

        direction_sums_repulse = direction_sums+repulse_scaled
        if verbose:
            print(f"Repulse = {repulse}.\n Repulse_norm = {repulse_norm}.\n Repulse_scaled ={repulse_scaled}")

        new_thetas = np.arctan2(direction_sums_repulse[:,1],direction_sums_repulse[:,0])+np.random.normal(0,sigma,N_alive)   
        self.positions[self.alive] = new_positions
        self.thetas[self.alive] = new_thetas
    

class Predator(Flock):
    def update_posdirs(self,dt,sigma,prey_class,attraction_factor =1,kill_radius=0.1,verbose=False):
        ##getting variables
        prey = prey_class.positions

        ##kill birds in kill radius
        dead = np.array(self.get_prey_in_radius(prey,kill_radius),dtype="bool")
        prey_class.kill(dead)

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
    def pvec_pp(self,prey):
        """Calculates the pairwise vectors between the prey and predators """
        N_prey = prey.shape[0]      
        tiled_prey = np.tile(prey,(self.N,1,1))
        tiled_predator =np.tile(self.positions,N_prey).reshape(tiled_prey.shape)   
        return tiled_prey-tiled_predator

    def get_prey_in_radius(self,prey,R):
        """Returns an N_pred by N_prey BOOL matrix, ith row representing which prey are in the ith 
        predators radius"""
        ##just returns prey to check update_posdir
        pvec =self.pvec_pp(prey)
        zeros =np.zeros(pvec.shape)
        L =self.frame_size
        pvec_wrapped =pvec -np.where(np.abs(pvec)>L/2,np.sign(pvec)*L,zeros)
        pdist_wrapped = lag.norm(pvec_wrapped,axis=2)
        indexs_mat =pdist_wrapped<R
        indexs =np.sum(indexs_mat,axis=0)
        return indexs
    
    def display_state_pp(self,prey):
        fig,ax = plt.subplots()
        fig.set_size_inches(14,8)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))

        ##get starting states
        pos_prey = prey.positions
        dir_prey =prey.get_directions()
        pos_pred = self.positions
        dir_pred =self.get_directions()

        ##plot the initial plot
        ax.quiver(pos_prey[:,0],pos_prey[:,1],dir_prey[:,0],dir_prey[:,1],scale = 100,color="b")
        ax.quiver(pos_pred[:,0],pos_pred[:,1],dir_pred[:,0],dir_pred[:,1],scale = 90,
        color="k",headwidth=5,minshaft=0.9)
    
    def animate_movement_pp(self,prey,frames,sigma,interval,repulsion_factor=1,attraction_factor=1):
        """Creates a quiver matplotlib animation of the birds moving """
        ##setting up plot
        fig,ax = plt.subplots()
        fig.set_size_inches(14,8)
        ax.set(xlim=(0,self.frame_size),ylim=(0,self.frame_size))

        ##get starting states
        init_pos_prey = prey.positions
        init_dir_prey =prey.get_directions()
        init_N_prey = init_pos_prey.shape[0]
        init_pos_pred = self.positions
        init_dir_pred =self.get_directions()

        ##plot the initial plot
        qprey =ax.quiver(init_pos_prey[:,0],init_pos_prey[:,1],init_dir_prey[:,0],init_dir_prey[:,1],scale = 100,color="b")
        qpred =ax.quiver(init_pos_pred[:,0],init_pos_pred[:,1],init_dir_pred[:,0],init_dir_pred[:,1],scale = 90,
        color="k",headwidth=5,minshaft=0.9)
        ##used in funcanimation to update states
        def animate(i):
            ##update birds positions 
            prey.update_posdirs(1,sigma,self.positions,repulsion_factor=repulsion_factor)
            self.update_posdirs(1,sigma,prey,attraction_factor=attraction_factor)

            ##update the quiver plots with new positions
            p_prey =prey.positions
            d_prey = prey.get_directions()
            p_pred = self.positions
            d_pred = self.get_directions()

            qprey.set_offsets(p_prey)
            qpred.set_offsets(p_pred)
            qprey.set_UVC(d_prey[:,0],d_prey[:,1])
            qpred.set_UVC(d_pred[:,0],d_pred[:,1])



        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames,blit= False
        )
        return anim
