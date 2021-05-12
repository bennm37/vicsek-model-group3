from flock import *
class Prey(Flock):


    def kill(self,dead):
        """Removes dead birds from the flock """
        alive =np.invert(dead)
        self.N = self.N-np.sum(dead)
        self.positions = self.positions[alive]
        self.thetas = self.thetas[alive]
        # print(f"killed birds: N={self.N},pos shape ={self.positions.shape}")


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
        self.positions = new_positions
        self.thetas = new_thetas
    

class Predator(Flock):
    def update_posdirs(self,dt,sigma,prey_class,attraction_factor =1,kill_radius=0.1,verbose=False):
        ##getting variables
        prey = prey_class.positions

        ##kill birds in kill radius
        dead = np.array(self.get_prey_in_radius(prey,kill_radius),dtype="bool")
        # print(dead)
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
            p_prey = prey.positions
            d_prey = prey.get_directions()
            p_pred = self.positions
            d_pred = self.get_directions()
            plt.clf()
            qprey =ax.quiver(init_pos_prey[:,0],init_pos_prey[:,1],init_dir_prey[:,0],init_dir_prey[:,1],scale = 100,color="b")
            qpred =ax.quiver(init_pos_pred[:,0],init_pos_pred[:,1],init_dir_pred[:,0],init_dir_pred[:,1],scale = 90,
            color="k",headwidth=5,minshaft=0.9)


        anim  = animation.FuncAnimation(
            fig,animate,interval = interval,frames =frames
        )
        return anim
