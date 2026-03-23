import csv
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch


class PostgresHandler:
    """Handles PostgreSQL connection and schema for training logs."""

    def __init__(self, db_config: dict):
        import psycopg2

        cfg = {k: v for k, v in db_config.items() if v != "" and v is not None}
        if "dsn" in cfg:
            self._conn = psycopg2.connect(cfg["dsn"])
        else:
            self._conn = psycopg2.connect(**cfg)
        self._init_schema()

    def _init_schema(self):
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) NOT NULL,
                    version VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    params JSONB,
                    metadata JSONB,
                    UNIQUE(model_name, version)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id SERIAL PRIMARY KEY,
                    run_id INTEGER REFERENCES training_runs(id) ON DELETE CASCADE,
                    epoch INTEGER NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    train_loss DOUBLE PRECISION NOT NULL,
                    val_loss DOUBLE PRECISION NOT NULL,
                    elapsed_time DOUBLE PRECISION
                )
            """)
        self._conn.commit()

    def upsert_run(self, model_name: str, version: str, metadata: dict = None) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_runs (model_name, version, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (model_name, version) DO UPDATE
                    SET metadata = EXCLUDED.metadata
                RETURNING id
                """,
                (model_name, version, json.dumps(metadata) if metadata else None),
            )
            run_id = cur.fetchone()[0]
        self._conn.commit()
        return run_id

    def update_params(self, run_id: int, params: dict):
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE training_runs SET params = %s WHERE id = %s",
                (json.dumps(params), run_id),
            )
        self._conn.commit()

    def insert_log(self, run_id: int, epoch: int, timestamp: str, train_loss: float, val_loss: float, elapsed_time=None):
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_logs (run_id, epoch, timestamp, train_loss, val_loss, elapsed_time)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (run_id, epoch, timestamp, float(train_loss), float(val_loss), elapsed_time),
            )
        self._conn.commit()

    def get_min_losses(self, run_id: int):
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(train_loss), MIN(val_loss) FROM training_logs WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return np.inf, np.inf
        return float(row[0]), float(row[1])

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model_path, model, optimizer, scheduler, best_loss, **kwargs):
    directory = os.path.dirname(model_path)
    os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            **kwargs,
        },
        model_path,
    )
    print(f"model checkpoint saved at {model_path}")


def load_model_params(model_folder, model_name, model_version, storage_handler=None):
    default_response = None
    params_file_name = f"{model_folder}/{model_name}_v{model_version}.json"
    if os.path.exists(params_file_name) is False:
        if storage_handler is None:
            print(f"exsisting model params not found on {params_file_name}.")
            return default_response
        else:
            response = storage_handler.download_file(f"/{model_name}/{model_name}_v{model_version}.json", params_file_name)
            if response is None:
                print("exsisting model params not found.")
                return default_response
    with open(params_file_name) as fp:
        params = json.load(fp)
    return params


def load_model(create_model_func, model_folder, model_name, model_version, storage_handler=None, device=None):
    if device is None:
        device = get_device()
    default_response = None, None
    params = load_model_params(model_folder, model_name, model_version, storage_handler)
    if params is None:
        return default_response

    model = create_model_func(**params).to(device)
    return params, model


def create_model_file_path(model_folder, model_name, model_version_str, is_train=False):
    if is_train:
        model_path = f"{model_folder}/{model_name}/{model_name}_train_{model_version_str}.torch"
    else:
        model_path = f"{model_folder}/{model_name}/{model_name}_{model_version_str}.torch"
    return model_path


def load_model_checkpoint(
    model,
    model_name,
    model_version_str,
    optimizer,
    scheduler,
    model_folder,
    train=True,
    storage_handler=None,
):
    default_response = (False, model, optimizer, scheduler, np.inf)
    model_path = create_model_file_path(model_folder, model_name, model_version_str, train)
    if os.path.exists(model_path) is False:
        if storage_handler is None:
            print("exsisting model not found.")
            return default_response
        file_name = os.path.basename(model_path)
        response = storage_handler.download_file(f"/{model_name}/{file_name}", model_path)
        if response is None:
            print("exsisting model not found.")
            return default_response

    if torch.cuda.is_available():
        check_point = torch.load(model_path)
    else:
        check_point = torch.load(model_path, map_location=torch.device("cpu"))
    if "model_state_dict" in check_point:
        model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        scheduler.load_state_dict(check_point["scheduler_state_dict"])
        if "best_loss" in check_point:
            best_loss = check_point["best_loss"]
        else:
            print("best_loss not found.")
            best_loss = np.inf
        print("state dict was loaded from checkpoint")
        return True, model, optimizer, scheduler, best_loss
    else:
        print("checkpoint is not available.")
        model.load_state_dict(check_point)
        return True, model, optimizer, scheduler, np.inf


def load_model_checkpoint_with_creation(
    create_model_func,
    model_name,
    model_version_str,
    model_folder,
    optimizer_class,
    scheduler_class,
    train=True,
    storage_handler=None,
    optimizer_kwargs={"lr": 1e-3},
    scheduler_kwargs={"step_size": 1, "gamma": 0.95},
):
    default_response = (False, None, None, None, None, np.inf)

    params, model = load_model(create_model_func, model_folder, model_name, model_version_str, storage_handler)
    if model is None:
        return default_response
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    scheduler = scheduler_class(optimizer, **scheduler_kwargs)
    succ, model, optimizer, scheduler, best_loss = load_model_checkpoint(
        model, model_name, model_version_str, optimizer, scheduler, model_folder, train, storage_handler
    )
    return succ, params, model, optimizer, scheduler, best_loss


class TrainingLogger:
    @classmethod
    def connect_drive(cls, mount_path="/content/drive"):
        from google.colab import drive

        drive.mount(mount_path)

    def __init__(self, model_name, version, base_path=None, storage_handler="colab", max_retry=3, local_cache_period=10, db_config=None, metadata=None):
        """Logging class to store training logs

        Args:
            model_name (str): It create a folder {base_path}/{model_name}/.
            verison (str): It create a file {base_path}/{model_name}/{model_name}_{version}.csv.
            base_path (str, optional): Base path to store logs. If you use cloud storage, this is used as temporal folder. Defaults to None.
            storage_handler (str|BaseHandler, optional): It change storage service. 'colab' can be selected. Defaults to 'colab'.
            max_retry (int, optional): max count of retry when store logs via network. Defaults to 3.
            local_cache_period(int, optional): Valid for cloud storage only. period to chache logs until send it to the storage. Defaults to 10.
        """
        # define common veriables
        self.MOUNT_PATH = "/content/drive"
        self.__use_cloud_storage = False
        self.__init_storage = lambda: None
        self.__local_cache_period = local_cache_period
        self.model_name = model_name
        self.version = version
        self.max_retry = max_retry

        # define variables depends on env
        if storage_handler == "colab":
            # this case we store logs on mounted path
            self.__init_colab()
            self.__init_storage = self.__init_colab
            if base_path is None:
                self.base_path = self.MOUNT_PATH
            else:
                base_pathes = [p for p in base_path.split("/") if len(p) > 0]
                self.base_path = os.path.join(self.MOUNT_PATH, "My Drive", *base_pathes)
        elif type(storage_handler) is str:
            raise ValueError(f"{storage_handler} is not supported. Please create StorageHandler for the service.")
        elif storage_handler is not None:
            # this case we store logs on app folder of dropbox, using cloud_storage_handlder
            self.__cloud_handler = storage_handler
            if self.__cloud_handler.refresh_token is None:
                self.__cloud_handler.authenticate()
            self.__use_cloud_storage = True
            if base_path is None:
                self.base_path = "./"
            else:
                self.base_path = base_path
        else:
            self.__cloud_handler = None
            if base_path is None:
                self.base_path = "./"
            else:
                self.base_path = base_path
        model_log_folder = os.path.join(self.base_path, model_name)
        if not os.path.exists(model_log_folder):
            os.makedirs(model_log_folder)
        file_name = f"{model_name}_{version}.csv"
        self.log_file_path = os.path.join(model_log_folder, file_name)
        self.__cache = []

        # PostgreSQL support
        self._db = None
        self._run_id = None
        self._db_epoch = 0
        if db_config is not None:
            try:
                self._db = PostgresHandler(db_config)
                self._run_id = self._db.upsert_run(model_name, version, metadata or {})
                print(f"PostgreSQL logger connected. run_id={self._run_id}")
            except Exception as e:
                print(f"Warning: Failed to connect to PostgreSQL: {e}. Falling back to CSV.")

    def __init_colab(self):
        from google.colab import drive

        drive.mount(self.MOUNT_PATH)

    def __store_files_to_cloud_storage(self, file_path):
        try:
            self.__cloud_handler.upload_training_results(self.model_name, [file_path])
            print("file uploaded to cloud storage ")
        except Exception as e:
            print(f"failed to save logs to cloud storage: {e}")

    def reset(self, model_name=None, file_name=None):
        if file_name is None:
            file_name = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if model_name is None:
            if file_name is None:
                raise ValueError("Either model_name or file_name should be specified")
            self.log_file_path = os.path.join(self.base_path, file_name)
        else:
            model_log_folder = os.path.join(self.base_path, model_name)
            if not os.path.exists(model_log_folder):
                os.makedirs(model_log_folder)
            self.log_file_path = os.path.join(model_log_folder, file_name)
        self.__cache = []

    def __cache_log(self, log_entry: list):
        self.__cache.append(log_entry)

    def __append_log(self, log_entry: list, retry_count=0):
        try:
            with open(self.log_file_path, "a", newline="") as log_file:
                writer = csv.writer(log_file)
                if len(self.__cache) > 0:
                    writer.writerows(self.__cache)
                    self.__cache = []
                writer.writerow(log_entry)
        except Exception as e:
            if retry_count < self.max_retry:
                if retry_count == 0:
                    print(e)
                self.__init_storage()
                self.__append_log(log_entry, retry_count + 1)
            else:
                self.__cache.append(log_entry)

    def save_params(self, params: dict, model_name=None, model_version=None):
        if self._db is not None and self._run_id is not None:
            try:
                params_for_db = params.copy()
                if "device" in params_for_db and not isinstance(params_for_db["device"], str):
                    params_for_db["device"] = str(params_for_db["device"])
                self._db.update_params(self._run_id, params_for_db)
            except Exception as e:
                print(f"Warning: Failed to save params to PostgreSQL: {e}")

        data_folder = os.path.dirname(self.log_file_path)
        if model_name is None:
            model_name = self.model_name
        if model_version is None:
            model_version = self.version
        if isinstance(model_version, int):
            param_file_path = os.path.join(data_folder, f"{model_name}_v{model_version}.json")
        else:
            param_file_path = os.path.join(data_folder, f"{model_name}_{model_version}.json")
        if "device" in params:
            device = params["device"]
            if not isinstance(device, str):
                params["device"] = str(device)
        with open(param_file_path, mode="w") as fp:
            json.dump(params, fp)
        if self.__use_cloud_storage:
            self.__store_files_to_cloud_storage(param_file_path)

    def save_model(self, model, model_name=None, model_version=None):
        if model is not None:
            data_folder = os.path.dirname(self.log_file_path)
            param_file_path = os.path.join(data_folder, f"{model_name}_v{model_version}.torch")
            torch.save(model.state_dict(), param_file_path)
            if self.__use_cloud_storage:
                self.__store_files_to_cloud_storage(param_file_path)

    def save_checkpoint(self, model, optimizer, scheduler, model_name, model_version, best_loss, **kwargs):
        if model is not None:
            data_folder = os.path.dirname(self.log_file_path)
            if isinstance(model_version, int):
                model_path = os.path.join(data_folder, f"{model_name}_v{model_version}.torch")
            else:
                model_path = os.path.join(data_folder, f"{model_name}_{model_version}.torch")
            save_checkpoint(model_path, model, optimizer, scheduler, best_loss, **kwargs)
            if self.__use_cloud_storage:
                self.__store_files_to_cloud_storage(model_path)

    def load_model_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        model_name=None,
        model_version=None,
        train=False,
        storage_handler=None,
        model_folder=None,
    ):
        if model_folder is None:
            data_folder = os.path.dirname(self.log_file_path)
        else:
            data_folder = model_folder
        if model_name is None:
            model_name = self.model_name
        if model_version is None:
            model_version = self.version
        return load_model_checkpoint(model, model_name, model_version, optimizer, scheduler, data_folder, train, storage_handler)

    def load_model_checkpoint_with_creation(
        self,
        create_model_func,
        optimizer_class,
        scheduler_class,
        model_name=None,
        model_version=None,
        train=False,
        storage_handler=None,
        optimizer_kwargs={"lr": 1e-3},
        scheduler_kwargs={"step_size": 1, "gamma": 0.95},
        model_folder=None,
    ):
        if model_folder is None:
            data_folder = os.path.dirname(self.log_file_path)
        else:
            data_folder = model_folder
        if model_name is None:
            model_name = self.model_name
        if model_version is None:
            model_version = self.version

        return load_model_checkpoint_with_creation(
            create_model_func,
            model_name,
            model_version,
            data_folder,
            optimizer_class,
            scheduler_class,
            train,
            storage_handler,
            optimizer_kwargs,
            scheduler_kwargs,
        )

    def save_logs(self):
        if self._db is not None and self._run_id is not None:
            self._db.close()
            return

        if len(self.__cache) > 0:
            with open(self.log_file_path, "a", newline="") as log_file:
                if len(self.__cache) > 0:
                    writer = csv.writer(log_file)
                    writer.writerows(self.__cache)
        if self.__use_cloud_storage:
            self.__store_files_to_cloud_storage(self.log_file_path)

    def add_training_log(self, training_loss, validation_loss, log_entry=None):
        timestamp = datetime.now().isoformat()

        if self._db is not None and self._run_id is not None:
            elapsed_time = log_entry if isinstance(log_entry, (int, float)) else None
            self._db_epoch += 1
            self._db.insert_log(self._run_id, self._db_epoch, timestamp, training_loss, validation_loss, elapsed_time)
            return

        basic_entry = [timestamp, training_loss, validation_loss]
        if log_entry is not None:
            if type(log_entry) is list and len(log_entry) > 0:
                basic_entry.extend(log_entry)
        if len(self.__cache) < self.__local_cache_period:
            self.__cache_log(basic_entry)
        else:
            self.__append_log(basic_entry)
            if self.__use_cloud_storage:
                self.__store_files_to_cloud_storage(self.log_file_path)

    def get_min_losses(self, train_loss_column=1, val_loss_column=2):
        if self._db is not None and self._run_id is not None:
            return self._db.get_min_losses(self._run_id)

        logs = None
        if os.path.exists(self.log_file_path) is False:
            if self.__cloud_handler is not None:
                file_name = os.path.basename(self.log_file_path)
                destination_path = f"/{self.model_name}/{file_name}"
                response = self.__cloud_handler.download_file(destination_path, self.log_file_path)
                if response is not None:
                    logs = pd.read_csv(self.log_file_path)
        else:
            try:
                logs = pd.read_csv(self.log_file_path)
            except pd.errors.EmptyDataError:
                logs = None

        if logs is None:
            print("no log available")
            return np.inf, np.inf
        else:
            if type(train_loss_column) is int:
                train_loss = logs.iloc[:, train_loss_column]
            elif type(train_loss_column) is str:
                train_loss = logs[train_loss_column]
            min_train_loss = train_loss.min()

            if type(val_loss_column) is int:
                val_loss = logs.iloc[:, val_loss_column]
            elif type(val_loss_column) is str:
                val_loss = logs[val_loss_column]
            min_val_loss = val_loss.min()

            return min_train_loss, min_val_loss
