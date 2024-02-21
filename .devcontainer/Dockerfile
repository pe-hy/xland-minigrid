FROM nvcr.io/nvidia/jax:23.08-py3

# TODO change
ARG USERNAME=pedro
# TODO change
ARG USER_EMAIL=pedro.dvoracek@gmail.com
ARG USER_UID=1000
ARG USER_GID=$USER_UID


COPY requirements.txt /tmp/
RUN apt-get update  \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    # create non-root user
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get install -y sudo tmux x11-apps \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # git configure
    && git config --global user.email "${USER_EMAIL}" \
    # git use main instead master branch name
    && git config --global init.defaultBranch main \
    # git avoid dubious ownership
    && git config --global --add safe.directory /workspaces/${WORKDIR_NAME}

# so zsh is installed for non-root user
USER $USERNAME
# install zsh
COPY zsh-in-docker.sh /tmp/zsh-in-docker.sh
RUN sh /tmp/zsh-in-docker.sh \
    -t amuse \
    -p git \
    -p extract \
    -p copybuffer
