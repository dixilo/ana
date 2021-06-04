import numpy as np
from scipy.constants import degree
import healpy as hp
import toast
try:
    import ephem
except:
    ephem = None
from toast import qarray as qa
from toast.timing import function_timer, Timer
from toast.tod import Interval, TOD
from toast.healpix import ang2vec
from toast.todmap.pointing_math import quat_equ2ecl, quat_equ2gal, quat_ecl2gal
from toast.todmap.sim_tod import simulate_hwp
from toast.weather import Weather

class TODGb(toast.todmap.TODGround):
    '''GB TOD
    '''
    @function_timer
    def __init__(
        self,
        mpicomm,
        detectors,
        samples,
        boresight_angle=0,
        firsttime=0.0,
        rate=100.0,
        site_lon=0,
        site_lat=0,
        site_alt=0,
        el=60,
        scanrate=1,
        scan_accel=0.1,
        CES_start=None,
        CES_stop=None,
        sun_angle_min=90,
        sampsizes=None,
        sampbreaks=None,
        coord="C",
        report_timing=True,
        hwprpm=None,
        hwpstep=None,
        hwpsteptime=None,
        sinc_modulation=False,
        **kwargs
    ):
        if samples < 1:
            raise RuntimeError(
                "TODGround must be instantiated with a positive number of "
                "samples, not samples == {}".format(samples)
            )

        if ephem is None:
            raise RuntimeError("Cannot instantiate a TODGround object without pyephem.")

        if sampsizes is not None or sampbreaks is not None:
            raise RuntimeError(
                "TODGround will synthesize the sizes to match the subscans."
            )

        if CES_start is None:
            CES_start = firsttime
        elif firsttime < CES_start:
            raise RuntimeError(
                "TODGround: firsttime < CES_start: {} < {}"
                "".format(firsttime, CES_start)
            )
        lasttime = firsttime + samples / rate
        if CES_stop is None:
            CES_stop = lasttime
        elif lasttime > CES_stop:
            raise RuntimeError(
                "TODGround: lasttime > CES_stop: {} > {}" "".format(lasttime, CES_stop)
            )

        self._firsttime = firsttime
        self._lasttime = lasttime
        self._rate = rate
        self._site_lon = site_lon
        self._site_lat = site_lat
        self._site_alt = site_alt

        if el < 1 or el > 89:
            raise RuntimeError("Impossible CES at {:.2f} degrees".format(el))

        self._boresight_angle = boresight_angle * degree
        self._el_ces = el * degree
        self._scanrate = scanrate * degree
        self._CES_start = CES_start
        self._CES_stop = CES_stop
        self._sun_angle_min = sun_angle_min
        if coord not in "CEG":
            raise RuntimeError("Unknown coordinate system: {}".format(coord))
        self._coord = coord
        self._report_timing = report_timing
        self._sinc_modulation = sinc_modulation

        self._observer = ephem.Observer()
        self._observer.lon = self._site_lon
        self._observer.lat = self._site_lat
        self._observer.elevation = self._site_alt  # In meters
        self._observer.epoch = ephem.J2000  # "2000"
        # self._observer.epoch = -9786 # EOD
        self._observer.compute_pressure()

        self._min_az = None
        self._max_az = None
        self._min_el = None
        self._min_el = None

        self._az = None
        self._commonflags = None
        self._boresight_azel = None
        self._boresight = None

        # Set the boresight pointing based on the given scan parameters

        tm = Timer()
        if self._report_timing:
            if mpicomm is not None:
                mpicomm.Barrier()
            tm.start()

        sizes, starts = self.simulate_scan(samples)

        if self._report_timing:
            if mpicomm is not None:
                mpicomm.Barrier()
            tm.stop()
            if (mpicomm is None) or (mpicomm.rank == 0):
                tm.report("TODGround: simulate scan")
            tm.clear()
            tm.start()

        self._fp = detectors
        self._detlist = sorted(list(self._fp.keys()))

        # call base class constructor to distribute data

        props = {
            "site_lon": site_lon,
            "site_lat": site_lat,
            "site_alt": site_alt,
            "el": el,
            "scanrate": scanrate,
            "scan_accel": scan_accel,
            "sun_angle_min": sun_angle_min,
        }
        super(toast.todmap.TODGround, self).__init__(
            mpicomm,
            self._detlist,
            samples,
            sampsizes=[samples],
            sampbreaks=None,
            meta=props,
            **kwargs
        )

        self._AU = 149597870.7
        self._radperday = 0.01720209895
        self._radpersec = self._radperday / 86400.0
        self._radinc = self._radpersec / self._rate
        self._earthspeed = self._radpersec * self._AU

        if self._report_timing:
            if mpicomm is not None:
                mpicomm.Barrier()
            tm.stop()
            if (mpicomm is None) or (mpicomm.rank == 0):
                tm.report("TODGround: call base class constructor")
            tm.clear()
            tm.start()

        self.translate_pointing()

        self.crop_vectors()

        if self._report_timing:
            if mpicomm is not None:
                mpicomm.Barrier()
            tm.stop()
            if (mpicomm is None) or (mpicomm.rank == 0):
                tm.report("TODGround: translate scan pointing")

        # If HWP parameters are specified, simulate and cache HWP angle

        simulate_hwp(self, hwprpm, hwpstep, hwpsteptime)

        return
    
    @function_timer
    def simulate_scan(self, samples):
        """ Simulate a constant elevation scan, either constant rate or
        1/sin(az)-modulated.

        """
        self._az = np.arange(samples)/self._rate*self._scanrate
        self._az %= 2 * np.pi
        self._el = np.full(samples, self._el_ces)

        self.update_scan_range(self._az, self._el)
        self._min_az = -np.pi
        self._max_az = np.pi

        self._commonflags = np.array([0]*samples, dtype=np.uint8)
        self._times = self._CES_start + np.arange(samples) / self._rate

        return [samples], [self._CES_start]