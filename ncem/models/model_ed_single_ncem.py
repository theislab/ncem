import tensorflow as tf

from ncem.models.layers import get_out, Decoder
from ncem.models.layers.single_gnn_layers import SingleMaxLayer, SingleGcnLayer, SingleGatLayer, SingleLrGatLayer


class ModelEd2Ncem:
    """Model class for NCEM encoder-decoder with graph layer IND (MAX) or GCN."""

    def __init__(
        self,
        input_shapes,
        latent_dim: tuple,
        dropout_rate: float,
        l2_coef: float,
        l1_coef: float,
        cond_type: str,
        use_type_cond: bool,
        use_domain: bool,
        scale_node_size: bool,
        output_layer: str,
        dec_intermediate_dim: int,
        dec_n_hidden: int,
        dec_dropout_rate: float,
        dec_l1_coef: float,
        dec_l2_coef: float,
        dec_use_batch_norm: bool,
        **kwargs,
    ):
        super().__init__()
        self.args = {
            "input_shapes": input_shapes,
            "latent_dim": latent_dim,
            "dropout_rate": dropout_rate,
            "l2_coef": l2_coef,
            "l1_coef": l1_coef,
            "use_domain": use_domain,
            "use_type_cond": use_type_cond,
            "scale_node_size": scale_node_size,
            "output_layer": output_layer,
            "dec_intermediate_dim": dec_intermediate_dim,
            "dec_n_hidden": dec_n_hidden,
            "dec_dropout_rate": dec_dropout_rate,
            "dec_l1_coef": dec_l1_coef,
            "dec_l2_coef": dec_l2_coef,
            "dec_use_batch_norm": dec_use_batch_norm,
        }
        in_lr_feature_dim = input_shapes[0]
        out_feature_dim = input_shapes[1]
        num_targets_dim = input_shapes[2]
        neighbors_dim = input_shapes[3]
        categ_condition_dim = input_shapes[4]
        domain_dim = input_shapes[5]

        # node features - node representation of neighbors: Input Tensor - shape=(None, targets, F-in)
        input_x_targets = tf.keras.Input(shape=(num_targets_dim, in_lr_feature_dim), name="node_features_targets")
        # node features - node representation of neighbors: Input Tensor - shape=(None, neighbors, F-in)
        input_x_neighbors = tf.keras.Input(shape=(num_targets_dim, neighbors_dim, in_lr_feature_dim),
                                           name="node_features_neighbors")
        # node size - reconstruction: Input Tensor - shape=(None, targets, 1)
        input_node_size = tf.keras.Input(shape=(num_targets_dim, 1), name="node_size_reconstruct")
        # adj_matrices - A: Input Tensor - shape=(None, targets, neighbors)
        input_a = tf.keras.Input(shape=(num_targets_dim, neighbors_dim), name="adjacency_matrix")
        # Categorical predictors: Input Tensor - shape=(None, targets, P)
        input_categ_condition = tf.keras.Input(shape=(num_targets_dim, categ_condition_dim),
                                               name="categorical_predictor")
        # domain information of graph - shape=(None, 1)
        input_g = tf.keras.layers.Input(shape=(domain_dim,), name="input_da_group", dtype="int32")

        if use_domain:
            categ_condition = tf.concat(
                [
                    input_categ_condition,
                    tf.tile(tf.expand_dims(tf.cast(input_g, dtype="float32"), axis=-2), [1, num_targets_dim, 1]),
                ],
                axis=-1,
            )
        else:
            categ_condition = input_categ_condition

        if cond_type == "lr_gat":
            x_encoder = SingleLrGatLayer(
                lr_dim=in_lr_feature_dim,
                dropout_rate=dropout_rate,
                l2_reg=l2_coef,
                name=f"lr_gat_layer",
            )([input_x_targets, input_x_neighbors, input_a])
        elif cond_type == "gat":
            x_encoder = SingleGatLayer(
                lr_dim=in_lr_feature_dim,
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                l2_reg=l2_coef,
                name=f"lr_gat_layer",
            )([input_x_targets, input_x_neighbors, input_a])
        elif cond_type == "max":
            print("MAX")
            x_encoder = SingleMaxLayer(
                name=f"max_layer"
            )([input_x_neighbors, input_a])
        elif cond_type == "gcn":
            x_encoder = SingleGcnLayer(
                latent_dim=latent_dim,
                dropout_rate=dropout_rate,
                activation="relu",
                l2_reg=l2_coef,
                use_bias=True,
                name=f"gcn_layer"
            )([input_x_targets, input_x_neighbors, input_a])
        elif cond_type == "none":
            x_encoder = input_x_targets
        else:
            raise ValueError("tried to access a non-supported conditional layer %s" % cond_type)

        # Decoder on neighborhood embeddings:
        if dec_n_hidden > 0:
            x = Decoder(
                intermediate_dim=dec_intermediate_dim,
                n_hidden=dec_n_hidden,
                dropout_rate=dec_dropout_rate,
                l1_coef=dec_l1_coef,
                l2_coef=dec_l2_coef,
                use_type_cond=use_type_cond,
                use_batch_norm=dec_use_batch_norm,
            )((x_encoder, categ_condition))
        else:
            x = tf.concat([x_encoder, categ_condition], axis=-1)

        output_decoder = get_out(output_layer=output_layer, out_feature_dim=out_feature_dim,
                                 scale_node_size=scale_node_size)((x, input_node_size))
        output_decoder_concat = tf.keras.layers.Concatenate(axis=2, name="reconstruction")(output_decoder)

        self.encoder = tf.keras.Model(
            inputs=[
                input_x_targets,
                input_x_neighbors,
                input_node_size,
                input_a,
                input_categ_condition,
                input_g,
            ],
            outputs=x_encoder,
            name="encoder_ncem",
        )
        self.training_model = tf.keras.Model(
            inputs=[
                input_x_targets,
                input_x_neighbors,
                input_node_size,
                input_a,
                input_categ_condition,
                input_g,
            ],
            outputs=output_decoder_concat,
            name="ed_ncem",
        )
