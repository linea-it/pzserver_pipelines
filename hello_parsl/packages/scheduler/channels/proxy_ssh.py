"""_summary_"""

import logging
import os

import paramiko
from parsl.channels import SSHChannel
from parsl.channels.errors import AuthException, BadHostKeyException, SSHException

logger = logging.getLogger(__name__)


class ProxySSHChannel(SSHChannel):
    """SSH persistent channel. This enables remote execution on sites
    accessible via ssh. It is assumed that the user has setup host keys
    so as to ssh to the remote host. Which goes to say that the following
    test on the commandline should work:

    >>> ssh <username>@<hostname>

    """

    def __init__(
        self,
        hostname,
        username=None,
        password=None,
        script_dir=None,
        envs=None,
        gssapi_auth=False,
        skip_auth=False,
        port=22,
        key_filename=None,
        host_keys_filename=None,
        ssh_config="~/.ssh/config",
    ):
        """Initialize a persistent connection to the remote system.
        We should know at this point whether ssh connectivity is possible

        Args:
            - hostname (String) : Hostname

        KWargs:
            - username (string) : Username on remote system
            - password (string) : Password for remote system
            - port : The port designated for the ssh connection. Default is 22.
            - script_dir (string) : Full path to a script dir where
                generated scripts could be sent to.
            - envs (dict) : A dictionary of environment variables to be set when
                executing commands
            - key_filename (string or list): the filename, or list of filenames,
                of optional private key(s)

        Raises:
        """

        super().__init__(
            hostname,
            username,
            password,
            script_dir,
            envs,
            gssapi_auth,
            skip_auth,
            port,
            key_filename,
            host_keys_filename,
        )

        self.ssh_config = ssh_config

        if not os.path.isfile(self.ssh_config):
            msg = (
                "Error: ssh_config not found. This class is useful for cases where "
                "you cannot reach your target node directly but have access to some "
                "gateway or staging box. And therefore, the ssh_config file is "
                "extremely necessary to create this tunnel via proxy. If your case "
                "doesn't need proxy, please use SSHChannel."
            )
            raise FileNotFoundError(msg)

        conf = paramiko.SSHConfig()
        with open(os.path.expanduser(self.ssh_config), encoding="utf-8") as sshconf:
            conf.parse(sshconf)
            host = conf.lookup(hostname)

        self.username = host.get("username", self.username)
        self.password = host.get("password", self.password)
        self.port = host.get("port", self.port)
        self.proxycommand = host.get("proxycommand")

    def _connect(self):
        if not self._is_connected():
            logger.debug("connecting to %s:%s", self.hostname, self.port)
            try:
                self.ssh_client.connect(
                    self.hostname,
                    username=self.username,
                    password=self.password,
                    port=self.port,
                    allow_agent=True,
                    gss_auth=self.gssapi_auth,
                    gss_kex=self.gssapi_auth,
                    key_filename=self.key_filename,
                    sock=paramiko.ProxyCommand(self.proxycommand),
                )
                transport = self.ssh_client.get_transport()
                self.sftp_client = paramiko.SFTPClient.from_transport(transport)

            except paramiko.BadHostKeyException as error:
                logger.error("hostname %s", self.hostname)
                raise BadHostKeyException from error

            except paramiko.AuthenticationException as error:
                logger.error("hostname %s", self.hostname)
                raise AuthException from error

            except paramiko.SSHException as error:
                logger.error("hostname %s", self.hostname)
                print("----> hostname %s", self.hostname)
                raise SSHException from error

            except Exception as error:
                logger.error("hostname %s", self.hostname)
                raise SSHException from error
