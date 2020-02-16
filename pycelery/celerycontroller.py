import asyncio
import subprocess
import time
from pycelery.tasks import shutdown_gazebo, run_gazebo, evaluate_robot, run_gazebo_and_analyzer, insert_robot
from pycelery.converter import args_to_dic, dic_to_args, dic_to_pop, pop_to_dic
from pyrevolve.custom_logging.logger import logger

class CeleryController:
    """
    This class handles requests to celery workers such as starting a process, evaluating a robot or shutting down a process or worker.
    Note that this class also functions as a simulator_queue.

    :param settings: The settings namespace of the experiment.
    """

    def __init__(self, settings):
        self.settings = settings
        self.settingsDir = args_to_dic(settings)

        # Start celery
        self.start_workers()

    def start_workers(self):
        """
        Starts n_cores celery workers
        """

        logger.info("Starting a worker at the background using " + str(self.settings.n_cores) + " cores. ")
        subprocess.Popen("celery multi restart "+str(self.settings.n_cores)+" -Q robots -A pycelery -P celery_pool_asyncio:TaskPool -c 1", shell=True)

    async def shutdown(self):
        """
        A function to call all celery workers and shut them down.
        """

        shutdowns = []
        for i in range(self.settings.n_cores):
            sd = await shutdown_gazebo.delay()
            shutdowns.append(sd)

        for i in range(self.settings.n_cores):
            await shutdowns[i].get()

        subprocess.Popen("pkill -9 -f 'celery worker'", shell=True)
        subprocess.Popen("pkill -9 -f 'gzserver'", shell=True)

    async def start_gazebo_instances(self):
        """
        This functions starts N_CORES number of gazebo instances.
        Every worker owns one gazebo instance.
        """
        startup = await insert_robot.apply_async(("Start simulator celery queue.",), serializer="json")

        gws = []
        grs = []
        for i in range(self.settings.n_cores):
            gw = await run_gazebo_and_analyzer.delay(self.settingsDir, i)
            gws.append(gw)

        print("Until here is fine!")
        print(await startup.get())
        print("After the get statement.")
        # Testing the last gw.
        for j in range(self.settings.n_cores):
            await gws[j].get()

    async def test_robot(self, robot, conf):
        """
        :param robot: robot (individual)
        :param conf: configuration of the experiment
        :return: future which contains fitness and measures.
        """

        # Create a yaml text from robot
        yaml_bot = robot.phenotype.to_yaml()

        future = await evaluate_robot_test.delay(yaml_bot, conf.fitness_function, self.settingsDir)

        # return the future
        return future
