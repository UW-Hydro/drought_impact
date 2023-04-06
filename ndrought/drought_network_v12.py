from multiprocessing import Event
import networkx as nx
import numpy as np
import matplotlib as mpl
import ndrought.wrangle as wrangle
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pickle
from datetime import datetime
import os

import ndrought.plotting as ndplot

import matplotlib.animation
from matplotlib.animation import FuncAnimation
plt.rcParams["animation.html"] = "html5"


class EventNode():

    def __init__(self, time, area, coords, id):
        """Singular drought blob at a specific time.

        Parameters
        ----------
        time
            Expected to be an integer currently. Further
            support for other time indices does not exist.
        area
            Calculated size of the drought event.
        coords
            List of coordinates the make up the drought event.
        id
            Integer id to identify this specific space-time blob.
        
        """
        self.time = time
        self.area = area
        self.coords = coords
        self.id = id
        # going to want a group_id to identify drought events
        self.group_id = id
        # contain all the futures to point to
        self.future: List[EventNode] = list()
        # and reference the pasts in case we want to crawl
        # backwards, or later use it to assign group_id
        self.past: List[EventNode] = list()

    def __str__(self):
        future_events = list()
        for future_EventNode in self.future:
            future_events.append(future_EventNode.id)
        return f'time: {self.time}, id: {self.id}, futures: {future_events}'

    def __iter__(self):
        yield self
        for node in self.future:
            yield node

    def __repr__(self):
        return f'time: {self.time}, id: {self.id}'

    def __eq__(self, other):
        if isinstance(other, EventNode):
            return(
                (self.id == other.id) and
                (self.area == other.area) and
                (np.array_equal(self.coords, other.coords)) and
                (self.group_id == other.group_id)
            )

    def append_future(self, other):
        """Adds a node to future.

        Parameters
        ----------
        self, other: EventNode
        
        """
        self.future.append(other)
        

    def check_connects(self, other, auto_connect=True):
        """Checks if coords are shared between two EventNode's.

        Parameters
        ----------
        self, other: EventNode
        auto_connect, (optional): boolean
            Whether to automatically append other to the future
            of self if found to be connected. Default set to True.

        Returns
        -------
        boolean
            Whether connection was found or not. Note that if
            auto_connect is True, nothing additional is returned
            but the EventNode self is modified.        
        """
        
        connection_found = False

        self_coord_set = set(tuple(coord) for coord in self.coords)
        other_coord_set = set(tuple(coord) for coord in other.coords)

        if len(self_coord_set.intersection(other_coord_set)) > 0:
            connection_found = True
            if auto_connect:
                self.append_future(other)
        
        return connection_found

    def get_future_thread(self, thread=None):
        """Gathers nodes that connect via future.

        Recursively crawls through future to collect
        all the nodes. Note that because this is a 
        recursive crawl, it may not lead to nodes
        being gathered in chronological order.

        Parameters
        ----------
        self: DroughtNetwork
        node: EventNode
        thread, (optional): list
            Contains futures found thus far. Passing
            this recursively eliminated redundancy.

        Returns
        -------
        list
            All EventNodes in the future of given EventNode.
        
        """

        if thread is None:
            thread = []

        if not self in thread:
            thread.append(self)
        
        if len(self.future) > 0:
            for future_node in self.future:
                future_node.get_future_thread(thread)

        return thread


def create_EventNodes(vals:np.ndarray, time=0, id=0, threshold=1, area_threshold=0):
    """Creates an EventNode if drought blob exists.

    While the EventNode and DroughtNetwork class are
    helpful for housing the drought blobs, we still need
    to transfer the data from wrangle.identify_drought_blob
    into the classes. That's where this function comes in.

    Parameters
    ----------
    vals: np.ndarray
        Binary drought array at a single time slice.
    time: int
        Time index for vals
    id: int    
        id to start labeling found drought blobs at.
        If more than one blob is found, then 1 is added
        to this id each time. For example, if given id
        0, and there were three blobs, they would be
        blobs 0, 1, and 2.

    Returns
    -------
    List[EventNode], int
        The list of created EventNode's for any blobs
        found, as well as what the next available id is.

    """
    df = wrangle.identify_drought_blob(vals, threshold)
    nodes = []
    for i in np.arange(len(df)):
        if df['area'].values[i] > area_threshold:
            node = EventNode(
                time=time,
                area=df['area'].values[i],
                coords=df['coords'].values[i],
                id=id
            )
            nodes.append(node)
            id += 1

    # hopeful optimization
    df = None

    return nodes, id

class DroughtNetwork:

    def __init__(self, data, threshold=1, area_threshold=0, name='drought_network'):
        """
        
        Parameters
        ----------
        data
            Expecting zeroth dimension to be temporal
            while the first and second dimension are
            spatial.
        """
        self.data = data.copy()
        data = self.data
        self.origins: List[EventNode] = list()
        self.nodes: List[EventNode] = list()
        self.threshold = threshold
        # record when this was made to improve record
        self.spawn_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        # some notes for legibility
        self.notes = "Put notes in here"
        self.name = name

        # go through and setup network
        last_nodes = []
        id = 0
        for i in tqdm(np.arange(data.shape[0]), desc=f'Creating Network: {name}'):
            nodes_i, id = create_EventNodes(data[i,:,:], time=i, id=id, threshold=threshold, area_threshold=area_threshold)
            # see if we currently found some droughts
            if len(nodes_i) > 0:
                # and if last time step there were droughts
                # that this might connect to;
                # also make sure to keep the list of all nodes
                # going strong
                self.nodes.extend(nodes_i)
                if len(last_nodes) > 0:
                    for node in nodes_i:
                        # loop through the last nodes 
                        # to see if there is a connection
                        connection_found = False
                        for last_node in last_nodes:
                            check = last_node.check_connects(node)
                            if check:                
                                connection_found = True
                                node.past.append(last_node)
                        
                        if not connection_found:
                            # if we couldn't find a connection
                            # then I'm considering it a new
                            # event for now
                            self.origins.append(node)

                else:
                    self.origins.extend(nodes_i)
                # need to preserve that we just had droughts
                # for the next time step
                last_nodes = nodes_i
            else:
                # if nothing was found, clear the holding list
                last_nodes = []

        # now that we have the nodes setup, let's 
        # assign group_id
        for node in self.nodes:
            # days of future past
            if len(node.past) == 1 and len(node.past[0].future) == 1:
                node.group_id = node.past[0].group_id

        # lastly, let's setup and adjacency matrix
        # the id will have been the last id plus 1,
        # which is great to set array dimensions
        adj_dict = dict()
        for node in self.nodes:
            id = node.id
            if id not in adj_dict.keys():
                adj_dict[id] = []
            for future in node.future:
                adj_dict[id].append(future.id)
        self.adj_dict = adj_dict
        
    def __eq__(self, other):
        if isinstance(other, DroughtNetwork):
            return(
                (self.threshold == other.threshold) and
                (self.origins == other.origins) and
                (self.nodes == other.nodes)
            )

    def __str__(self):
        return f"{self.name}: DroughtNetwork with D{self.threshold} threshold, spawned {self.spawn_time}"

    def __iter__(self):
        for node in self.nodes:
            yield node
    
    def find_node_by_id(self, id):
        try:
            found_node = self.nodes[id]
            # in case the nodes are no longer in order,
            # do a linear crawl
            if found_node.id != id:
                for node in self.nodes:
                    if id == node.id:
                        found_node = node 

            return found_node
        except:
            raise Exception(f'id: {id} not in network')

    
            
    def get_chronological_future_thread(self, id):
        """Collects future thread in chronological order.

        While this uses get_future_thread, it
        also sorts the nodes into chronological order
        to make plotting and the like easier that the
        recursive function does not originally do.

        Parameters
        ----------
        id: int
            What id to gather thread from.

        Returns
        -------
        List[EventNode]
        
        """
        node = self.find_node_by_id(id)
        nodes_to_sort = node.get_future_thread().copy()
        sorted_nodes = [nodes_to_sort[0]]
 
        if len(nodes_to_sort) > 1:
            for node in nodes_to_sort[1:]:
                if node.time >= sorted_nodes[-1].time:
                    sorted_nodes.append(node)
                else:
                    i = 0
                    while i <= len(sorted_nodes) and node.time > sorted_nodes[i].time:
                        i += 1
                    sorted_nodes.insert(i, node)

        # toss a little check in here
        if len(nodes_to_sort) != len(sorted_nodes):
            raise Exception('Something went wrong during sorting, check code.')

        return sorted_nodes

    def time_slice(self, start_time=None, end_time=None, id=None):
        """Collect nodes within network between certain times.

        Parameters
        ----------
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.

        Returns
        -------
        List[EventNode]

        """
        if id:
            # an option to combine the features of the two
            # functions
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes

        if start_time is None:
            start_time = self.nodes[0].time
        if end_time is None:
            end_time = self.nodes[-1].time
        
        time_sliced = []
        for node in nodes:
            t = node.time
            if t >= start_time and t <= end_time:
                time_sliced.append(node)
        
        return time_sliced

    def get_nx_network(self, id=None, start_time=None, end_time=None, adj_dict=None):
        """Gets topography and positions for networkx.

        Used for plotting in networkx.draw_networkx

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        adj_dict, (optional): dict
            Use an alternative adjacency dictionary to the
            whole network. Defaults to using whole network if None
            is given. This overrules selecting by id, start_time, or
            end_time.

        Returns
        -------
        topog, pos
            Positions generated from the following function:
            nx.drawing.nx_agraph.graphviz_layout(topog, prog= 'dot')
        
        """
        if adj_dict is None:
            if start_time or end_time:
                plot_nodes = self.time_slice(start_time, end_time, id)
                
            elif isinstance(id, int):
                plot_nodes = self.get_chronological_future_thread(id)
            else:
                plot_nodes = self.nodes

            plot_ids = [node.id for node in plot_nodes]
            adj_dict = self.filter_adj_dict_by_id(plot_ids)

        
        topog = nx.Graph(adj_dict)      
        pos = self.get_nx_pos(topog)

        return topog, pos

    def get_nx_pos(self, topog):
        return nx.drawing.nx_agraph.graphviz_layout(topog, prog= 'dot')

    def thread_timeseries(self, id=None, start_time=None, end_time=None):
        """Get time series from thread.

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.

        Returns
        -------
        time, vals: np.ndarray
            Time indices and corresponding values for timeseries.
        
        """

        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif isinstance(id, int):
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes

        start_time_found = nodes[0].time
        end_time_found = nodes[-1].time

        time = np.arange(start_time_found, end_time_found+1, 1)
        vals = np.zeros(len(time))

        for node in nodes:
            vals[time==node.time] += node.area

        return time, vals

    def stacked_events_plot(self, id=None, start_time=None, end_time=None, 
    ax=None, plot_legend=False, cmap=plt.cm.get_cmap('hsv'), **kwargs):
        """Generates a stacked plot of droughts.
        
        Parameters
        ----------
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        ax, (optional)
            matplotlib.pyplot axis object. If None given, then
            one is created.
        plot_legend, (optional): boolean
            Whether to plot the legend (True) or not (False).
            Defaults to True. Labels by group_id.
        cmap, (optional)
            Colormap to code group_id.
        """
        
        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif isinstance(id, int):
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes

        found_start_time = nodes[0].time
        found_end_time = nodes[-1].time

        time = np.arange(found_start_time, found_end_time+1, 1)
        template = np.zeros(len(time))
        groupings = dict()

        for node in nodes:
            if node.group_id not in groupings.keys():
                groupings[node.group_id] = template.copy()

            groupings[node.group_id][time == node.time] += node.area

        grouped_events = [groupings[key] for key in groupings.keys()]
        color_array = np.linspace(0, 1, len(grouped_events))
        colors = cmap(color_array)

        
        if ax is None:
            __, ax = plt.subplots()

        ax.stackplot(
            time,
            *grouped_events,
            labels=[f'{key}' for key in groupings.keys()],
            colors=colors,
            **kwargs
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Area in Drought Event')

        if plot_legend:
            ax.legend()

        return ax
    
    def to_array(self, id=None, start_time=None, end_time=None, adj_dict=None):
        """Converts the network into a 3-D array.

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        adj_dict, (optional): dict
            Use an alternative adjacency dictionary to the
            whole network. Defaults to using whole network if None
            is given.

        Returns
        -------
        np.array
            First dimension is time, second and third are lat/lon. The
            array is only binary values based no the threshold originally
            given to the network.

        """

        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif isinstance(id, int):
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes
        
        if adj_dict:
            temp = []
            for node in nodes:
                if node.id in adj_dict.keys():
                    temp.append(node)
            nodes = temp

        times = [node.time for node in nodes]

        found_start_time = np.array(times).min()
        found_end_time = np.array(times).max()

        array_out = np.zeros(((found_end_time-found_start_time)+1, self.data.shape[1], self.data.shape[2]))

        for node in nodes:
            t = node.time - found_start_time
            for [i, j] in node.coords:
                array_out[t, i, j] += 1
        
        return array_out
                
    def temporal_color_map(self, start_time=None, end_time=None, id=None, adj_dict=None, 
    cmap=plt.cm.get_cmap('hsv')):
        """Creates a node_color color map by temporal value.

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        adj_dict, (optional): dict
            Use an alternative adjacency dictionary to the
            whole network. Defaults to using whole network if None
            is given.

        Returns
        -------
        list
        """
        
        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif isinstance(id, int):
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes

        if adj_dict:
            temp = []
            for node in nodes:
                if node.id in adj_dict.keys():
                    temp.append(node)
            nodes = temp

        times = [node.time for node in nodes]

        found_start_time = np.array(times).min()
        found_end_time = np.array(times).max()
        time_range = found_end_time - found_start_time
        #print(times)

        color_map = []

        for time in times:
            time_percent = (time-found_start_time)/time_range
            color_map.append(time_percent)

        return color_map
        

    def relative_area_color_map(self, start_time=None, end_time=None, id=None, adj_dict=None, 
    cmap=plt.cm.get_cmap('hsv')):
        """Creates a node_color color map by area value.

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.
        start_time, (optional): int
            What time to start the slice at, inclusive. If None
            is given, then the time of the first node in the
            DroughtNetwork is used.
        end_time, (optional): int
            What time to end the slice at, inclusive. If None is
            given, then the time of the last node in the 
            DroughtNetwork is used.
        adj_dict, (optional): dict
            Use an alternative adjacency dictionary to the
            whole network. Defaults to using whole network if None
            is given.

        Returns
        -------
        list
        """

        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif isinstance(id, int):
            nodes = self.get_chronological_future_thread(id)
        else:
            nodes = self.nodes

        if adj_dict:
            temp = []
            for node in nodes:
                if node.id in adj_dict.keys():
                    temp.append(node)
            nodes = temp

        areas = [node.area for node in nodes]
        max_area = np.max(areas)

        color_map = []

        for area in areas:
            area_percent = area/max_area
            color_map.append(cmap(area_percent)[:-1])
        
        return color_map

    def pickle(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    def unpickle(path):
        with open(path, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            return unpickler.load()

    def filter_adj_dict_by_area(self, area_filter):
        """Filters adjacency dictionary by area.

        Parameters
        ----------
        area_filter: int/float
            Numeric value to filter area by. Only areas
            greater than this value will remain.
        
        Returns
        -------
        dict
        """

        adj_dict_filtered = dict()

        for node in self.nodes:
            if node.area > area_filter:
                id = node.id
                if id not in adj_dict_filtered.keys():
                    adj_dict_filtered[id] = []
                for future in node.future:
                    if future.area > area_filter:
                        adj_dict_filtered[id].append(future.id)

        return adj_dict_filtered

    def filter_adj_dict_by_id(self, id_filter):
        """Filter adjacency dictionary by id.

        Parameters
        ----------
        id_filter: array-like
            id values to be retained in the adjacency
            dictionary returned.
        
        Returns
        -------
        dict
        """

        adj_dict_filtered = dict()

        for id in id_filter:
            adj_dict_vals = self.adj_dict[id]
            adj_dict_vals_filtered = []
            for val in adj_dict_vals:
                if val in id_filter:
                    adj_dict_vals_filtered.append(val)
            adj_dict_filtered[id] = adj_dict_vals_filtered

        return adj_dict_filtered
    
    def create_animated_gif(self, out_path:str, adj_dict=None, fps=2, overwrite=False, times=None, nonbinary_data=None):
        """Create and save an animated gif of drought evolution.

        Parameters
        ----------
        out_path: str
            Where to save the animation to. Note that this function
            does not output an animation to the terminal nor jupyter
            cell, only to this saved location.
        adj_dict, (optional): dict
            Adjacency dictionary for selecting specific nodes in the
            DroughtNetwork object. Default is to assume use of the
            entire network.
        fps, (optional): int
            Frames per second to run the animation at, defaults to 2 fps.
        overwrite, (optional): boolean
            Whether to enable overwriting of the animation should the
            file already exist. Defaults to False to prevent accidental
            overwriting.
        times, (optional): array-like
            Animation frame titles, typically just the actual date since
            the DroughtNetwork does not store time itself, only an
            index of time.
        nonbinary_data, (optional): np.ndarray
            While the DroughtNetwork data only stores a binary
            representation of drought based on the provided threshold,
            this allows the user to pass in nonbinary data, (USDM
            category expected based on color-coding), to use the
            binary data as a mask and provide a more detailed animation.
        """
        fig, ax = plt.subplots()
        
        if adj_dict is None:
            adj_dict = self.adj_dict
        animate_array = self.to_array(adj_dict=adj_dict)
        if not nonbinary_data is None:
            #assert isinstance(nonbinary_data, np.ndarray)
            #if animate_array.shape != nonbinary_data.shape:
            #    raise Exception("nonbinary_data does not match the size of this network's data")

            mask = np.ma.make_mask(animate_array==0)
            animate_array = np.ma.masked_where(mask, nonbinary_data)
        frames = animate_array.shape[0]
        
        if times is None:
            times = np.arange(frames)

        if not nonbinary_data is None:
            cmap = ndplot.usdm_cmap()
            def animate(i):
                ax.clear()
                ax.invert_yaxis()
                ax.set_title(times[i])
                ax.set_facecolor('k')
                return (ax.pcolormesh(animate_array[i, :, :], cmap=cmap, vmin=-1, vmax=4),)
        else:
        
            def animate(i):
                ax.clear()
                ax.invert_yaxis()
                ax.set_title(times[i])
                return (ax.pcolormesh(animate_array[i, :, :]),)
        
        ani = mpl.animation.FuncAnimation(fig, animate, frames=frames, blit=True)
        writer = mpl.animation.ImageMagickWriter(fps=fps)
        if os.path.exists(out_path) and not overwrite:
            raise Exception('File already exists. Please delete or enable overwrite.')
        else:
            try:
                os.remove(out_path)
            except:
                pass
            ani.save(out_path, writer=writer)
        plt.close()


    def find_overlapping_nodes_events(self, other, matched_dates_dict_idx:dict):
        """Find overlapping nodes with another DroughtNetwork.

        Parameters
        ----------
        other: DroughtNetwork
        matched_dates_dict_idx: dict
            Map going from the time indices in self DroughtNetwork as
            keys and other DroughtNetwork as values.
        
        Returns
        -------
        list[dict]
            List of event threads that overlap between self and other,
            where each element in the list is a dictionary mapping
            what nodes are overlapping between the two DroughtNetworks
            with the times of self as keys.

            [[{self.time:[[self_overlapping_node, other_overlapping_node], ...]}], ...]

        """

        overlapped_nodes = dict()

        # this can take a bit, so will make a progress bar
        t = tqdm(total=len(self.nodes)*len(other.nodes), desc=f'Overlapping {self.name} & {other.name}')

        for node_self in self.nodes:
            # first need to see if the node's time is a matched time
            time_idx = node_self.time
            if time_idx in matched_dates_dict_idx.keys():
                # will be testing overlap via set intersection
                node_self_coord_set = set(tuple(coord) for coord in node_self.coords)
                # now we need to find what time we are looking for
                # in other that's matched up to the time we're looking at
                matched_idx = matched_dates_dict_idx[time_idx]
                for node_other in other.nodes:
                    # if it's a temporal match
                    if matched_idx == node_other.time:
                        # then we'll carry through testing for spatial intersection
                        node_other_coord_set = set(tuple(coord) for coord in node_other.coords)                        
                        if len(node_self_coord_set.intersection(node_other_coord_set)) > 0:
                            if time_idx not in overlapped_nodes.keys():
                                overlapped_nodes[time_idx] = []
                            overlapped_nodes[time_idx].append([node_self, node_other])

                    t.update()

        overlap_events = []

        # okay, now to figure out which are the temporally consecutive
        # events to have delineations between overlaps
        current_event = []
        for idx in overlapped_nodes.keys():
            # if we currently aren't constructing an event,
            # then we must be starting from scratch and will
            # just toss it on to get started
            if len(current_event) == 0:
                current_event.append({idx:overlapped_nodes[idx]})
            # now we want to see if they're consecutive
            elif list(current_event[-1].keys())[0] == idx -1:
                current_event.append({idx:overlapped_nodes[idx]})
            # if they aren't consecutive, then we need to
            # store the current event we were working on
            # and start from scratch
            else:
                overlap_events.append(current_event)
                current_event = [{idx:overlapped_nodes[idx]}]
        
        # lastly, if the final time is consecutive, then we
        # won't end up storing in our events, so we should
        # check whether it got stored and store it if not
        if len(current_event) != 0 and overlap_events[-1] != current_event:
            overlap_events.append(current_event)


        return overlap_events        
    
    def area_thresh_removal(self, thresh:int):
        to_remove = []
        for node in tqdm(self.nodes, desc='Searching'):
            if node.area <= thresh:
                to_remove.append(node)
        
        for node in tqdm(to_remove, desc='Removing'):
            for past_node in node.past:
                past_node.future.remove(node)
            for future_node in node.future:
                future_node.past.remove(node)
            self.nodes.remove(node)
            if node in self.origins:
                self.origins.remove(node)

    def area_thresh_splice(self, thresh):
        t = tqdm(total=len(self.nodes))
        for node in self.nodes:
            t.set_description('Searching')
            to_splice = []
            for future_node in node.future:
                if future_node.area <= thresh*node.area:
                    to_splice.append(future_node)
            
            for splice_node in to_splice:
                t.set_description('Splicing')
                node.future.remove(splice_node)
                splice_node.past.remove(node)
                if len(splice_node.past) == 0:
                    self.origins.append(splice_node)
            t.update()
        
def compute_alignment_fraction(overlap_events):

    net_af = []

    for thread in overlap_events:
        thread_af = dict()

        for event in thread:
            time = list(event.keys())[0]

            event_a = np.array(event[time])[:, 0]
            coords_a = np.vstack([node.coords for node in event_a])
            coord_set_a = set(tuple(coord) for coord in coords_a)

            event_b = np.array(event[time])[:, 1]
            coords_b = np.vstack([node.coords for node in event_b])
            coord_set_b = set(tuple(coord) for coord in coords_b)

            coord_set_intersect = coord_set_a.intersection(coord_set_b)
            coord_set_union = coord_set_a.union(coord_set_b)

            thread_af[time] = len(coord_set_intersect)/len(coord_set_union)
    
        net_af.append(thread_af)

    return net_af

def compute_total_alignment_fraction(overlap_events):

    intersect_total = 0
    union_total = 0

    for thread in overlap_events:
        for event in thread:
            time = list(event.keys())[0]

            event_a = np.array(event[time])[:, 0]
            coords_a = np.vstack([node.coords for node in event_a])
            coord_set_a = set(tuple(coord) for coord in coords_a)

            event_b = np.array(event[time])[:, 1]
            coords_b = np.vstack([node.coords for node in event_b])
            coord_set_b = set(tuple(coord) for coord in coords_b)

            coord_set_intersect = coord_set_a.intersection(coord_set_b)
            coord_set_union = coord_set_a.union(coord_set_b)

            intersect_total += len(coord_set_intersect)
            union_total += len(coord_set_union)
    
    return intersect_total/union_total

def compute_disagreement_fraction(a_net, b_net, overlap_events):

    a_overlapped = dict()
    b_overlapped = dict()
    times = []

    for thread in overlap_events:
        for event in thread: 
            time = list(event.keys())[0]

            if not time in a_overlapped.keys():
                a_overlapped[time] = []
            if not time in b_overlapped.keys():
                b_overlapped[time] = []

            event_a = np.array(event[time])[:, 0]
            event_b = np.array(event[time])[:, 1]
            times.append(time)

            a_overlapped[time].extend(np.hstack(event_a))
            b_overlapped[time].extend(np.hstack(event_b))

    a_nodes = dict()
    b_nodes = dict()

    a_not_overlapped = dict()
    b_not_overlapped = dict()

    for node in a_net.nodes:
        time = node.time
        
        if not time in a_nodes.keys():
            a_nodes[time] = []
        a_nodes[time].append(node)

        if not time in a_overlapped.keys() or not node in a_overlapped[time]:
            if time not in a_not_overlapped.keys():
                a_not_overlapped[time] = []
            a_not_overlapped[time].append(node)

    for node in b_net.nodes:
        time = node.time

        if not time in b_nodes.keys():
            b_nodes[time] = []
        b_nodes[time].append(node)

        if not time in b_overlapped.keys() or not node in b_overlapped[time]:
            if time not in b_not_overlapped.keys():
                b_not_overlapped[time] = []
            b_not_overlapped[time].append(node)

    a_df = dict()
    b_df = dict()

    for time in a_not_overlapped.keys():
        all_nodes = a_nodes[time]
        not_overlapped_nodes = a_not_overlapped[time]

        total_area = 0
        not_overlapped_area = 0

        for node in not_overlapped_nodes:
            not_overlapped_area += len(node.coords)
        for node in all_nodes:
            total_area += len(node.coords)

        a_df[time] = not_overlapped_area/total_area

    for time in b_not_overlapped.keys():
        all_nodes = b_nodes[time]
        not_overlapped_nodes = b_not_overlapped[time]

        total_area = 0
        not_overlapped_area = 0

        for node in not_overlapped_nodes:
            not_overlapped_area += len(node.coords)
        for node in all_nodes:
            total_area += len(node.coords)

        b_df[time] = not_overlapped_area/total_area

    return a_df, b_df

def compute_total_disagreement_fraction(a_net, b_net, overlap_events):

    a_overlapped = dict()
    b_overlapped = dict()
    times = []

    for thread in overlap_events:
        for event in thread: 
            time = list(event.keys())[0]

            if not time in a_overlapped.keys():
                a_overlapped[time] = []
            if not time in b_overlapped.keys():
                b_overlapped[time] = []

            event_a = np.array(event[time])[:, 0]
            event_b = np.array(event[time])[:, 1]
            times.append(time)

            a_overlapped[time].extend(np.hstack(event_a))
            b_overlapped[time].extend(np.hstack(event_b))

    a_nodes = dict()
    b_nodes = dict()

    a_not_overlapped = dict()
    b_not_overlapped = dict()

    for node in a_net.nodes:
        time = node.time
        
        if not time in a_nodes.keys():
            a_nodes[time] = []
        a_nodes[time].append(node)

        if not time in a_overlapped.keys() or not node in a_overlapped[time]:
            if time not in a_not_overlapped.keys():
                a_not_overlapped[time] = []
            a_not_overlapped[time].append(node)

    for node in b_net.nodes:
        time = node.time

        if not time in b_nodes.keys():
            b_nodes[time] = []
        b_nodes[time].append(node)

        if not time in b_overlapped.keys() or not node in b_overlapped[time]:
            if time not in b_not_overlapped.keys():
                b_not_overlapped[time] = []
            b_not_overlapped[time].append(node)

    a_not_overlapped_total = 0
    a_area_total = 0

    b_not_overlapped_total = 0
    b_area_total = 0

    for time in a_not_overlapped.keys():
        all_nodes = a_nodes[time]
        not_overlapped_nodes = a_not_overlapped[time]

        total_area = 0
        not_overlapped_area = 0

        for node in not_overlapped_nodes:
            not_overlapped_area += len(node.coords)
        for node in all_nodes:
            total_area += len(node.coords)

        a_not_overlapped_total += not_overlapped_area
        a_area_total += total_area

    for time in b_not_overlapped.keys():
        all_nodes = b_nodes[time]
        not_overlapped_nodes = b_not_overlapped[time]

        total_area = 0
        not_overlapped_area = 0

        for node in not_overlapped_nodes:
            not_overlapped_area += len(node.coords)
        for node in all_nodes:
            total_area += len(node.coords)

        b_not_overlapped_total += not_overlapped_area
        b_area_total += total_area

    return a_not_overlapped_total/a_area_total, b_not_overlapped_total/b_area_total



