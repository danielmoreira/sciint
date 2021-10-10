# Basis image: dmoreira/faiss
FROM dmoreira/faiss

# Copies the solution to docker.
COPY ./ /ranking
WORKDIR /ranking
RUN chmod 775 /ranking/01_query_all_images.sh
RUN chmod 775 /ranking/02_eval_all_ranks.sh

# Creates the input-output folder
# (it must be mounted as a volume when executing this container).
RUN mkdir -p /ranking/io
