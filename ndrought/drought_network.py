import networkx as nx
import numpy as np
import ndrought.wrangle as wrangle
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

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

    def get_future_thread(self, futures=[]):
        """Gathers nodes that connect via future.

        Recursively crawls through future to collect
        all the nodes. Note that because this is a 
        recursive crawl, it may not lead to nodes
        being gathered in chronological order.

        Parameters
        ----------
        self: EventNode
        futures, (optional): list
            Contains futures found thus far. Passing
            this recursively eliminated redundancy.

        Returns
        -------
        list
            All EventNodes in the future of given EventNode.
        
        """

        if not self in futures:
            futures.append(self)
       
        if len(self.future) > 0:
            for node in self.future:
                futures = node.get_future_thread(futures)
        
        return futures

def create_EventNodes(vals:np.ndarray, time=0, id=0):
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
    df = wrangle.identify_drought_blob(vals)
    nodes = []
    for i in np.arange(len(df)):
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

    def __init__(self, data):
        """
        
        Parameters
        ----------
        data
            Expecting zeroth dimension to be temporal
            while the first and second dimension are
            spatial.
        """
        self.data = data
        self.origins: List[EventNode] = list()
        self.nodes: List[EventNode] = list()

        # go through and setup network
        last_nodes = []
        id = 0
        for i in tqdm(np.arange(data.shape[0])):
            nodes_i, id = create_EventNodes(data[i,:,:], time=i, id=id)
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
        self.adj_mat = np.zeros((id, id))
        for node in self.nodes:
            i_id = node.id
            for future in node.future:
                j_id = future.id
                self.adj_mat[i_id, j_id] = 1
        
    def find_node_by_id(self, id)  :
        for node in self.nodes:
            if node.id == id:
                return node
        
        raise Exception(f'id: {id} not in network')
            
    def get_chronological_future_thread(self, id):
        """Collects future thread in chronological order.

        While this uses EventNode.get_future_thread, it
        also sorts the nodes into chronological order
        to make plotting and the like easier that the
        recursive function does not originally do.

        Parameters
        ----------
        id: int
            What id to gather thread from.

        Returns
        -------
        List[EventNode
        
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

    def get_nx_network(self, id=None):
        """Gets topography and positions for networkx.

        Used for plotting in networkx.draw_networkx

        Parameters
        ----------
        id, (optional): int
            If you would like to select out an id thread at the
            same time, this can be given to use 
            get_chronological_future_thread to time slice through.

        Returns
        -------
        topog, pos
            Positions generated from the following function:
            nx.drawing.nx_agraph.graphviz_layout(topog, prog= 'dot')
        
        """
        topog = nx.from_numpy_array(self.adj_mat)

        if id is None:
            plot_nodes = self.nodes
        else:
            plot_nodes = self.get_chronological_future_thread(id)

        plot_ids = [node.id for node in plot_nodes]
        
        if plot_nodes != self.nodes:
            for node in self.nodes:
                if node.id not in plot_ids:
                    topog.remove_node(node.id)
        
        pos = nx.drawing.nx_agraph.graphviz_layout(topog, prog= 'dot')

        return topog, pos

    def thread_timeseries(self, id=None, start_time=None, end_time=None):
        """Get time series from thread.

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
        time, vals: np.ndarray
            Time indices and corresponding values for timeseries.
        
        """

        if start_time or end_time:
            nodes = self.time_slice(start_time, end_time, id)
        elif id:
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
    ax=None, plot_legend=False, cmap=plt.cm.get_cmap('hsv')):
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
        elif id:
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
            colors=colors
        )
        ax.set_xlabel('Time')
        ax.set_ylabel('Area in Drought Event')

        if plot_legend:
            ax.legend()

        return ax
    
